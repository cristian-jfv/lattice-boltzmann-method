#ifndef PARAMS_HPP
#define PARAMS_HPP
#include <iostream>
#include <cmath>

struct dimensional_params
{
	double nu{1.0533e-6};
	double c{343};
	double U{0.001};
	double L{0.01};
	double rho_0{1.0};
	double p_0{101325.0};
	double t_0 = L/c;
};

std::ostream& operator<<(std::ostream& os, const dimensional_params& p);

struct dimensionless_params
{
	double tau;
	double Re;
	double nu;
	double dx;
	int H,W;
	dimensionless_params(const dimensional_params &p)
	{
		tau = 0.55;
		Re = p.U*p.L/p.nu;
		nu = p.nu/(p.c*p.L);
		dx = 3.0*nu/(tau - 0.5);
		H = ceil(p.L/dx);
		W = ceil(p.L/dx);
	}
};

std::ostream& operator<<(std::ostream& os, const dimensionless_params& p);

struct lattice_units{

};
#endif
