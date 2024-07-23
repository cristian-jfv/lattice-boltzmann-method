#include <c10/core/DeviceType.h>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <ostream>
#include <cassert>
#include <string>
#include <torch/torch.h>
#include <torch/types.h>

#include "../src/utils.hpp"
#include "../src/solver.hpp"
#include "../src/ulbm.hpp"

using std::cout;
using std::endl;
using torch::Tensor;
using torch::indexing::None;
using torch::indexing::Slice;
using torch::indexing::Ellipsis;
using utils::print;
using utils::indices;

// Boundaries
const indices virtual_inlet{0, Ellipsis};
const indices inlet{1, Ellipsis};
const indices outlet{-2, Ellipsis};
const indices virtual_outlet{-1, Ellipsis};

const torch::Tensor E = torch::tensor(
  {4.0/ 9.0,
    1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

const torch::Tensor c = torch::tensor(
  {{0.0, 1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
   {0.0, 0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0}},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));


void set_initial_conditions
(
  Tensor& rho,
  Tensor& u,
  const double u_max,
  const double alpha=80.0,
  const double delta=0.05
)
{
  const int R = u.size(0);
  const int C = u.size(1);

  for (int r = 0; r < R; r++)
  {
    for (int c = 0; c < C; c++)
    {
      u.index({r,c,0}) = u_max*std::tanh(alpha*(0.25*R - std::abs(c - 0.5*R)));
      rho.index({r,c}) = 1.0; //-0.001*u_max*std::sin(6.2832*r/R)*std::sin(6.28322*c/R);
      u.index({r,c,1}) = u_max*delta*std::sin(6.2832*(r+0.25*R)/R);
    }
  }
}

int main()
{
  // Flow parameters
  const int T = 10000;
  const int snapshot_period = 10;
  print("T", T);
  const int H = 256;
  const int W = 256;
  cout << "H=" << H << "; W=" << W << endl;

  const double nu = 3.413333E-4;
  const double omega = 1.0/(0.5 + 3.0*nu);
  const double tau = 1.0/omega;
  print("nu", nu);
  print("omega", omega);
  print("tau", tau);
  const double u_max = 0.02;
  print("u_max", u_max);
  print("Re", W*u_max/nu);

  torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(torch::kDouble));
  cout << "Default torch dtype: " << torch::get_default_dtype() << endl;
  if (!torch::cuda::is_available())
  {
    std::cerr << "CUDA is NOT available\n";
  }

  ulbm::d2q9::kbc kbc{H,W,omega};
  // Initial conditions
  kbc.m0.fill_(1.0);
  set_initial_conditions(kbc.m0, kbc.m1, u_max);
  kbc.eval_equilibrium(kbc.adve_f);

  // Results
  const int Ts = T/snapshot_period;
  //Tensor fs = torch::zeros({H,W,9,Ts});
  Tensor ux = torch::zeros({H,W,Ts});
  Tensor uy = torch::zeros({H,W,Ts});
  Tensor rhos = torch::zeros({H,W,Ts});

  // Main loop
  print("main loop starts");
  for (int t=0; t < T; t++)
  {
    if (t%snapshot_period == 0)
    {
      cout << t << "\t\r" << std::flush;
      int ts = t/snapshot_period;
      //fs.index_put_({Ellipsis,t}, kbc.adve_f);
      ux.index_put_({Ellipsis,ts}, kbc.m1.index({Ellipsis,0}));
      uy.index_put_({Ellipsis,ts}, kbc.m1.index({Ellipsis,1}));
      rhos.index_put_({Ellipsis,ts}, kbc.m0.squeeze(-1));
    }

    kbc.collide();
    kbc.advect();

    // Periodic boundary conditions
    // top
    kbc.adve_f.index_put_({0,Slice(0,-1),8}, kbc.coll_f.index({-1, Slice(1,None), 8}));
    kbc.adve_f.index_put_({0,Slice(),1}, kbc.coll_f.index({-1, Slice(), 1}));
    kbc.adve_f.index_put_({0,Slice(1,None),5}, kbc.coll_f.index({-1, Slice(0,-1), 5}));
    // bottom
    kbc.adve_f.index_put_({-1,Slice(0,-1),7}, kbc.coll_f.index({0, Slice(1,None), 7}));
    kbc.adve_f.index_put_({-1,Slice(),3}, kbc.coll_f.index({0, Slice(), 3}));
    kbc.adve_f.index_put_({-1,Slice(1,None),6}, kbc.coll_f.index({0, Slice(0,-1), 6}));
    // left
    kbc.adve_f.index_put_({Slice(0,-1),0,6}, kbc.coll_f.index({Slice(1,None),-1,6}));
    kbc.adve_f.index_put_({Slice(),0,2}, kbc.coll_f.index({Slice(),-1,2}));
    kbc.adve_f.index_put_({Slice(1,None),0,5}, kbc.coll_f.index({Slice(0,-1),-1,5}));
    // right
    kbc.adve_f.index_put_({Slice(0,-1),-1,7}, kbc.coll_f.index({Slice(1,None),0,7}));
    kbc.adve_f.index_put_({Slice(),-1,4}, kbc.coll_f.index({Slice(),0,4}));
    kbc.adve_f.index_put_({Slice(1,None),-1,8}, kbc.coll_f.index({Slice(0,-1),0,8}));

    // Calculate macroscopic variables
    kbc.m0 = kbc.adve_f.sum(-1).detach().clone();
    kbc.m1 = (torch::matmul(kbc.adve_f,c.transpose(0,1))/kbc.m0.unsqueeze(-1)).detach().clone();
  }

  // Save results
  print("saving results into files");
  std::string file_prefix = "ulbm-double-shear-flow";
  torch::save(ux,   file_prefix + "-ux.pt");
  torch::save(uy,   file_prefix + "-uy.pt");
  //torch::save(fs,   file_prefix + "hpt-fs.pt");
  torch::save(rhos, file_prefix + "-rho.pt");

  return 0;
}
