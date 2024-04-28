# include "solver.hpp"
#include <c10/core/DefaultDtype.h>
#include <iostream>

namespace solver
{
using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

const torch::Tensor E = torch::tensor(
  {4.0/ 9.0,
    1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0}, torch::TensorOptions().dtype(torch::kDouble));
const torch::Tensor c = torch::tensor(
  {{0.0, 1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
   {0.0, 0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0}}, torch::TensorOptions().dtype(torch::kDouble));

void calc_rho(torch::Tensor& rho, const torch::Tensor& f)
{
    rho = f.sum_to_size(rho.sizes()).clone().detach();
}

void calc_incomp_u(torch::Tensor& u, const torch::Tensor& f)
{
  u = (matmul(f, c.transpose(0,1))).clone().detach();
}


void calc_u(torch::Tensor& u, const torch::Tensor& f, const torch::Tensor& rho)
{
  u = (matmul(f, c.transpose(0,1))/rho).clone().detach();
}

void incomp_equilibrium
(
  torch::Tensor &f_eq,
  const torch::Tensor &u,
  const torch::Tensor &rho
)
{
  auto c_u = matmul(u, c);
  auto A = rho + 3.0*c_u;
  f_eq = mul(A,E).clone().detach();
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

void advect(torch::Tensor& g, const torch::Tensor& f)
{
  // Advect
  // f0
  g.index({Ellipsis}) = f.index({Ellipsis}).clone().detach();

  // f1
  //print("f1");
  g.index({Slice(1,None), Slice(), 1}) = f.index({Slice(0,-1), Slice(), 1}).clone().detach();
  g.index({0, Slice(), 1}) = f.index({-1, Slice(), 1}).clone().detach();

  // f2
  //print("f2");
  g.index({Slice(), Slice(1,None), 2}) = f.index({Slice(), Slice(0,-1), 2}).clone().detach();
  g.index({Slice(), 0, 2}) = f.index({Slice(), -1, 2}).clone().detach();

  // f3
  //print("f3");
  g.index({Slice(0,-1), Slice(), 3}) = f.index({Slice(1,None), Slice(), 3}).clone().detach();
  g.index({-1, Slice(), 3}) = f.index({0,Slice(),3}).clone().detach();

  // f4
  //print("f4");
  g.index({Slice(), Slice(0,-1), 4}) = f.index({Slice(), Slice(1,None), 4}).clone().detach();
  g.index({Slice(), -1, 4}) = f.index({Slice(), 0, 4}).clone().detach();

  // f5
  //print("f5");
  g.index({Slice(1,None), Slice(1,None), 5}) = f.index({Slice(0,-1), Slice(0,-1),5}).clone().detach();
  g.index({0, Slice(1,None), 5}) = f.index({-1, Slice(0,-1), 5}).clone().detach();
  g.index({Slice(1,None), 0, 5}) = f.index({Slice(0,-1), -1, 5}).clone().detach();
  g.index({0, 0, 5}) = f.index({-1, -1, 5}).clone().detach();

  // f6
  //print("f6");
  g.index({Slice(0,-1), Slice(1,None), 6}) = f.index({Slice(1,None), Slice(0,-1), 6}).clone().detach();
  g.index({-1, Slice(1,None), 6}) = f.index({0, Slice(0,-1), 6}).clone().detach();
  g.index({Slice(0,-1), 0, 6}) = f.index({Slice(1,None), -1, 6}).clone().detach();
  g.index({-1, 0, 6}) = f.index({0, -1, 6}).clone().detach();

  // f7
  //print("f7");
  g.index({Slice(0,-1), Slice(0,-1), 7}) = f.index({Slice(1,None), Slice(1,None), 7}).clone().detach();
  g.index({-1, Slice(0,-1), 7}) = f.index({0, Slice(1,None), 7}).clone().detach();
  g.index({Slice(0,-1), -1, 7}) = f.index({Slice(1,None), 0, 7}).clone().detach();
  g.index({-1, -1, 7}) = f.index({0, 0, 7}).clone().detach();

  // f8
  //print("f8");
  g.index({Slice(1,None), Slice(0,-1), 8}) = f.index({Slice(0,-1), Slice(1,None), 8}).clone().detach();
  g.index({0, Slice(0,-1), 8}) = f.index({-1, Slice(1,None), 8});
  g.index({Slice(1,None), -1, 8}) = f.index({Slice(0,-1), 0, 8});
  g.index({0, -1, 8}) = f.index({-1, 0, 8}).clone().detach();

  //print("end of advect function");
}

}
