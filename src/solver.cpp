#include <iostream>
# include "solver.hpp"
using std::cout;
using std::endl;

const torch::Tensor E = torch::tensor({4.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 
                        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0});
const torch::Tensor c = torch::tensor(
	{{0.0, 1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
	 {0.0, 0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0}});

void initialize(double rho_0, double p_0,
		torch::Tensor &f,
		torch::Tensor &u,
		torch::Tensor &p)
{
	// Fill f, u, and p with initial values
	u =torch::zeros_like(u);
	p = (rho_0/3.0)*torch::ones_like(p);
	cout << "init f" << endl;
	f_eq(f, u, p);
}

void f_eq(torch::Tensor &f_eq,
	  const torch::Tensor &u,
	  const torch::Tensor &p)
{
	//cout << "E.sizes=" << E.sizes() << endl;
	//cout << "c.sizes=" << c.sizes() << endl;
	auto u_u = (u*u).sum_to_size(p.sizes());
	//cout << "u_u.sizes=" << u_u.sizes() << endl;
	auto c_u = matmul(u, c);
	f_eq = mul(3.0*p + 3.0*c_u + 4.5*c_u.pow(2) - 1.5*u_u, E);
}
