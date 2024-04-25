#ifndef SOLVER_HPP
#define SOLVER_HPP
#include <torch/torch.h>

namespace solver
{
void calc_rho(torch::Tensor& rho, const torch::Tensor& f);

void calc_u(torch::Tensor& u, const torch::Tensor& f, const torch::Tensor& rho);

void collision
(
  torch::Tensor& f_coll,
  const torch::Tensor& f_curr,
  const torch::Tensor& f_equi,
  const double omega
);

void equilibrium
(
  torch::Tensor &f_eq,
  const torch::Tensor &u,
  const torch::Tensor &rho
);

void advect(torch::Tensor& g, const torch::Tensor& f);

}
#endif
