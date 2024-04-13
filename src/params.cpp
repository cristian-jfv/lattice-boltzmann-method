#include "params.hpp"
#include <ostream>

std::ostream& operator<<(std::ostream& os, const dimensional_params& p)
{
	return os << "Dimensional parameters: " << std::endl
	 << "nu=" << p.nu << " m2/s" << std::endl
	 << "c=" << p.c <<  " m/s" << std::endl
	 << "U=" << p.U << " m/s" << std::endl
	 << "L=" << p.L << " m" << std::endl
	 << "rho_0=" << p.rho_0 << " kg/m3" << std::endl
	<< "p_0=" << p.p_0 << "Pa" << std::endl
	 << "t_0=" << p.t_0 << " s" << std::endl;
}

std::ostream& operator<<(std::ostream& os, const dimensionless_params& p)
{
	return os << "Dimensionless parameters:" << std::endl
	 << "Re=" << p.Re << std::endl
	 << "tau=" << p.tau << std::endl
	 << "nu=" << p.nu << std::endl
	 << "dx=" << p.dx << std::endl
	 << "H=" << p.H << std::endl
	 << "W=" << p.W << std::endl;
}
