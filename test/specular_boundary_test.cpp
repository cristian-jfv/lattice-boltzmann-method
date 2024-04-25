#include <ATen/TensorIndexing.h>
#include <ATen/core/Formatting.h>
#include <ATen/ops/ones_like.h>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <ostream>
#include <string>
#include <torch/serialize.h>
#include <torch/torch.h>

using std::cout;
using std::endl;
using torch::Tensor;
using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;


typedef std::initializer_list<at::indexing::TensorIndex> indices;
// Boundaries
const indices virtual_inlet{0, Ellipsis};
const indices inlet{1, Ellipsis};
const indices outlet{-2, Ellipsis};
const indices virtual_outlet{-1, Ellipsis};

const torch::Tensor E = torch::tensor(
  {4.0/ 9.0,
    1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0});
const torch::Tensor c = torch::tensor(
  {{0.0, 1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
   {0.0, 0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0}});

void store_tensor(const torch::Tensor& source, torch::Tensor& target, int i)
{
  using torch::indexing::Slice;
    // Save results
    target.index({Slice(), Slice(), Slice(), i}) =
      source.index({Slice(), Slice(), Slice()}).clone().detach();
}

void print(const std::string& str)
{
  cout << str << endl;
}

template<typename T>
void print(const std::string& name, const T& t)
{
  cout << name << "=" << t << endl;
}


void calc_u(torch::Tensor& u, const torch::Tensor& f, const torch::Tensor& rho)
{
  u = (matmul(f, c.transpose(0,1))/rho).clone().detach();
}


void equilibrium
(
  torch::Tensor &f_eq,
  const torch::Tensor &u,
  const torch::Tensor &rho
)
{
  auto u_u = (u*u).sum_to_size(rho.sizes());
  auto c_u = matmul(u, c);
  auto A = 1.0 + 3.0*c_u + 4.5*c_u.pow(2) - 1.5*u_u;
  f_eq = mul(rho*A, E).clone().detach();
}


void collision
(
  torch::Tensor& f_coll,
  const torch::Tensor& f_curr,
  const torch::Tensor& f_equi,
  const double omega
)
{
  f_coll = ( (1.0-omega)*f_curr + omega*f_equi ).clone().detach();
}


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
  torch::Tensor temp_equi = torch::zeros({1, u.sizes()[1], 9});
  torch::Tensor temp_rho = torch::ones({1, u.sizes()[1], 1});

  // inlet
  equilibrium(temp_equi, u.index(outlet).unsqueeze(0), rho_inlet*temp_rho);
  f_coll.index(virtual_inlet) = (temp_equi + f_coll.index(outlet) - f_equi.index(outlet)).squeeze(0).clone().detach();

  // outlet
  equilibrium(temp_equi, u.index(inlet).unsqueeze(0), rho_outlet*temp_rho);
  f_coll.index(virtual_outlet) = (temp_equi + f_coll.index(inlet) - f_equi.index(inlet)).squeeze(0).clone().detach();
}


void advect(torch::Tensor& f_adve, const torch::Tensor& f_coll)
{
  // Advect
  // f1
  f_adve.index({Slice(), Slice(), 1-1}) =
    f_coll.index({Slice(), Slice(), 1-1}).clone().detach();

  // f2
  f_adve.index({Slice(), Slice(1,None), 2-1}) =
    f_coll.index({Slice(), Slice(0,-1), 2-1}).clone().detach();
//  // f2 carry
//  f_adve.index({Slice(), 0, 2-1}) =
//    f_curr.index({Slice(), 0, 2-1}).clone().detach();

  // f3
  f_adve.index({Slice(0,-1), Slice(), 3-1}) =
    f_coll.index({Slice(1,None), Slice(), 3-1}).clone().detach();
//  // f3 carry
//  f_adve.index({-1, Slice(), 3-1}) =
//    f_curr.index({-1, Slice(), 3-1}).clone().detach();

  // f4
  f_adve.index({Slice(), Slice(0,-1), 4-1}) =
    f_coll.index({Slice(), Slice(1,None), 4-1}).clone().detach();
//  // f4 carry
//  f_adve.index({Slice(), -1, 4-1}) =
//    f_curr.index({Slice(), -1, 4-1}).clone().detach();

  // f5
  f_adve.index({Slice(1,None), Slice(), 5-1}) =
    f_coll.index({Slice(0,-1), Slice(), 5-1}).clone().detach();
//  // f5 carry
//  f_adve.index({0, Slice(), 5-1}) =
//    f_curr.index({0, Slice(), 5-1}).clone().detach();

  // f6
  f_adve.index({Slice(0,-1), Slice(1,None), 6-1}) =
    f_coll.index({Slice(1,None), Slice(0,-1), 6-1}).clone().detach();
//  // f6 carry
//  f_adve.index({Slice(), 0, 6-1}) =
//    f_curr.index({Slice(), 0, 6-1}).clone().detach();
//  f_adve.index({-1, Slice(), 6-1}) =
//    f_curr.index({-1, Slice(), 6-1}).clone().detach();


  // f7
  f_adve.index({Slice(0,-1), Slice(0,-1), 7-1}) =
    f_coll.index({Slice(1,None), Slice(1,None), 7-1}).clone().detach();
//  f_adve.index({Slice(), -1, 7-1}) =
//    f_curr.index({Slice(), -1, 7-1}).clone().detach();
//  f_adve.index({-1, Slice(), 7-1}) =
//    f_curr.index({-1, Slice(), 7-1}).clone().detach();

  // f8
  f_adve.index({Slice(1,None), Slice(0,-1), 8-1}) =
    f_coll.index({Slice(0,-1), Slice(1,None), 8-1}).clone().detach();
//  f_adve.index({Slice(), -1, 8-1}) =
//    f_curr.index({Slice(), -1, 8-1}).clone().detach();
//  f_adve.index({0, Slice(), 8-1}) =
//    f_curr.index({0, Slice(), 8-1}).clone().detach();

  // f9
  f_adve.index({Slice(1,None), Slice(1,None), 9-1}) =
    f_coll.index({Slice(0,-1), Slice(0,-1), 9-1}).clone().detach();
//  f_adve.index({0, Slice(), 9-1}) =
//    f_curr.index({0, Slice(), 9-1}).clone().detach();
//  f_adve.index({Slice(), 0, 9-1}) =
//    f_curr.index({Slice(), 0, 9-1}).clone().detach();
}

int main()
{
  // Flow parameters
  const int T = 5000;
  print("T", T);
  const int H = 11;
  const int W = 11;
  cout << "H=" << H << "; W=" << W << endl;
  const double tau = std::sqrt(3.0/16.0) + 0.5;
  const double omega = 1.0/tau;
  cout << "omega=" << omega << endl;
  const double u_max = 0.1;
  const double nu = (2.0 * tau -1.0)/6.0;
  cout << "nu=" << nu << endl;
  const double Re = W*u_max/nu;
  cout << "Re=" << Re << endl;
  const double p_grad = 8.0*nu*u_max/(W*W);
  cout << "grad(p)=" << p_grad << endl;
  const double rho_outlet = 1.0;
  const double rho_inlet = 3.0*(H-1)*p_grad + rho_outlet;
  cout << "rho_inlet=" << rho_inlet << endl;

  // Tensors
  Tensor f_equi = torch::zeros({H,W,9});
  Tensor f_coll = torch::zeros_like(f_equi);
  Tensor f_adve = torch::zeros_like(f_equi);
  Tensor u = torch::zeros({H,W,2});
  Tensor rho = torch::ones({H,W,1});

  // Results
  Tensor fs = torch::zeros({H,W,9,T});
  Tensor ux = torch::zeros({H,W,T});
  Tensor uy = torch::zeros({H,W,T});
  Tensor rhos = torch::zeros({H,W,T});
  //cout << u << endl;
  print("u", u);

  // Initialisation
  equilibrium(f_adve, u, rho);

  // Main loop
  print("main loop starts");
  for (int t=0; t < T; t++)
  {
    cout << t << "\t\r" << std::flush;
    //print("save fs");
    fs.index_put_({Ellipsis,t}, f_adve);
    //print("save ux");
    ux.index_put_({Ellipsis,t}, u.index({Ellipsis,0}));
    //print("save uy");
    uy.index_put_({Ellipsis,t}, u.index({Ellipsis,1}));
    //print("save rho");
    rhos.index_put_({Ellipsis,t}, rho.squeeze(2));

    // Calculate macroscopic variables
    //print("calculate macroscopic variables");
    rho = f_adve.sum_to_size(rho.sizes()).clone().detach();
    calc_u(u, f_adve, rho);
    // Compute equilibrium
    //print("compute equilibrium");
    equilibrium(f_equi, u, rho);
    // Collision, BGK operator
    //print("collision step");
    collision(f_coll, f_adve, f_equi, omega);
    // Inlet and outlet periodic boundary conditions
    //print("inlet-outlet priodic boundary condition");
    periodic_boundary_condition(f_coll, f_equi, u, rho, rho_inlet, rho_outlet);
    // Advection
    //print("advection step");
    //advect(f_adve, f_coll);

    for (int k=0; k<9; k++)
    {
      //print("*********************************************");
      //print("k", k);
      for (int j=0; j<W; j++)
      {
        for (int i=0; i<H; i++)
        {
          int newx = (i+c.index({0,k}).item().toInt()+H) % H;
          int newy = (j+c.index({1,k}).item().toInt()+W) % W;
          //cout << "i=" << i << "; j=" << j << endl;
          //cout << "newx=" << newx << "; newy=" << newy << endl;
          f_adve.index({newx, newy, k}) = f_coll.index({i,j,k}).clone().detach();
        }
      }
    }

    // No slip boundary conditions
    f_adve.index({Slice(), -1, 4}) = f_coll.index({Slice(), -1, 2});
    f_adve.index({Slice(), -1, 7}) = f_coll.index({Slice(), -1, 5});
    f_adve.index({Slice(), -1, 8}) = f_coll.index({Slice(), -1, 6});

    f_adve.index({Slice(), 0, 2}) = f_coll.index({Slice(), 0, 4});
    f_adve.index({Slice(), 0, 5}) = f_coll.index({Slice(), 0, 7});
    f_adve.index({Slice(), 0, 6}) = f_coll.index({Slice(), 0, 8});
  }

  // Save results
  print("saving results into files");
  torch::save(ux, "sbt-ux.pt");
  torch::save(uy, "sbt-uy.pt");
  torch::save(fs, "sbt-fs.pt");
  torch::save(rhos, "sbt-rho.pt");
  return 0;
}

