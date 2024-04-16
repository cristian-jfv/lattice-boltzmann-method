#ifndef SOLVER_HPP
#define SOLVER_HPP
#include <torch/torch.h>

void initialize(double rho_0, double p_0,
		torch::Tensor& f,
		torch::Tensor& u,
		torch::Tensor& p);

void f_eq(torch::Tensor& f_eq,
	  const torch::Tensor& u,
	  const torch::Tensor& p);

void f_step(torch::Tensor& f_next,
	    const torch::Tensor& f_curr,
	    const torch::Tensor& f_eq,
	    double eps);

void p(torch::Tensor& p,
       const torch::Tensor& f);

void u(torch::Tensor& u,
       const torch::Tensor& f);
#endif
