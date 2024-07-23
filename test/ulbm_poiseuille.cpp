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


void periodic_boundary_condition
(
  torch::Tensor& f_coll,
  const torch::Tensor& f_equi,
  const torch::Tensor& u,
  const torch::Tensor& rho,
  const double rho_inlet,
  const double rho_outlet
)
{
  torch::Tensor temp_equi = torch::zeros({1, u.sizes()[1], 9}, torch::kCUDA);
  torch::Tensor temp_rho = torch::ones({1, u.sizes()[1], 1}, torch::kCUDA);

  // inlet
  solver::incomp_equilibrium(temp_equi, u.index(outlet).unsqueeze(0), rho_inlet*temp_rho);
  f_coll.index(virtual_inlet) = (temp_equi + f_coll.index(outlet) - f_equi.index(outlet)).squeeze(0).clone().detach();

  // outlet
  solver::incomp_equilibrium(temp_equi, u.index(inlet).unsqueeze(0), rho_outlet*temp_rho);
  f_coll.index(virtual_outlet) = (temp_equi + f_coll.index(inlet) - f_equi.index(inlet)).squeeze(0).clone().detach();
}

int main()
{
  // Flow parameters
  const int T = 300000;
  const int snapshot_period = 100;
  print("T", T);
  const int H = 128;
  const int W = 128;
  cout << "H=" << H << "; W=" << W << endl;

  const double nu = 1E-4;
  const double omega = 1.0/(0.5 + 3.0*nu);
  const double tau = 1.0/omega;
  print("nu", nu);
  print("omega", omega);
  print("tau", tau);
  const double u_max = 0.05;
  print("u_max", u_max);
  print("Re", W*u_max/nu);

  const double p_grad = 8.0*nu*u_max/(W*W);
  cout << "grad(p)=" << p_grad << endl;
  const double rho_outlet = 1.0;
  const double rho_inlet = 3.0*(H-1)*p_grad + rho_outlet;
  cout << "rho_inlet=" << rho_inlet << endl;


  torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(torch::kDouble));
  cout << "Default torch dtype: " << torch::get_default_dtype() << endl;
  if (!torch::cuda::is_available())
  {
    std::cerr << "CUDA is NOT available\n";
  }

  ulbm::d2q9::kbc kbc{H,W,omega};
  kbc.m0.fill_(1.0);

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
    periodic_boundary_condition(kbc.coll_f, kbc.iequi_f.pow(-1), kbc.m1, kbc.m0, rho_inlet, rho_outlet);
    kbc.advect();

    // No slip boundary conditions for the first and last column
    kbc.adve_f.index({Slice(), -1, 4}) = kbc.coll_f.index({Slice(), -1, 2}).clone().detach();
    kbc.adve_f.index({Slice(), -1, 7}) = kbc.coll_f.index({Slice(), -1, 5}).clone().detach();
    kbc.adve_f.index({Slice(), -1, 8}) = kbc.coll_f.index({Slice(), -1, 6}).clone().detach();

    kbc.adve_f.index({Slice(),  0, 2}) =  kbc.coll_f.index({Slice(), 0, 4}).clone().detach();
    kbc.adve_f.index({Slice(),  0, 5}) =  kbc.coll_f.index({Slice(), 0, 7}).clone().detach();
    kbc.adve_f.index({Slice(),  0, 6}) =  kbc.coll_f.index({Slice(), 0, 8}).clone().detach();

    // No slip boundary conditions for the first and last rows
/*    kbc.adve_f.index({ -1,Slice(1,-1), 3}) = kbc.coll_f.index({ -1,Slice(1,-1), 1}).clone().detach();
    kbc.adve_f.index({ -1,Slice(1,-1), 6}) = kbc.coll_f.index({ -1,Slice(1,-1), 8}).clone().detach();
    kbc.adve_f.index({ -1,Slice(1,-1), 7}) = kbc.coll_f.index({ -1,Slice(1,-1), 5}).clone().detach();

    kbc.adve_f.index({  0,Slice(1,-1), 1}) = kbc.coll_f.index({  0,Slice(1,-1), 3}).clone().detach();
    kbc.adve_f.index({  0,Slice(1,-1), 5}) = kbc.coll_f.index({  0,Slice(1,-1), 7}).clone().detach();
    kbc.adve_f.index({  0,Slice(1,-1), 8}) = kbc.coll_f.index({  0,Slice(1,-1), 6}).clone().detach();
*/

    // Calculate macroscopic variables
    // solver::calc_rho(kbc.m0, kbc.adve_f);
    kbc.m0 = kbc.adve_f.sum(-1).detach().clone();
    // solver::calc_incomp_u(kbc.m1, kbc.adve_f);
    kbc.m1 = (torch::matmul(kbc.adve_f,c.transpose(0,1))/kbc.m0.unsqueeze(-1)).detach().clone();
  }

  // Save results
  print("saving results into files");
  std::string file_prefix = "ulbm-poiseuille";
  torch::save(ux,   file_prefix + "hpt-ux.pt");
  torch::save(uy,   file_prefix + "hpt-uy.pt");
  //torch::save(fs,   file_prefix + "hpt-fs.pt");
  torch::save(rhos, file_prefix + "hpt-rho.pt");

  return 0;
}
