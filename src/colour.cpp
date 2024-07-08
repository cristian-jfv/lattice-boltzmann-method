#include <c10/core/DeviceType.h>
#include <iostream>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <toml++/toml.hpp>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "colour.hpp"

colour::colour(const toml::node_view<toml::node>& tbl, int rows, int cols, const torch::Tensor& E):
rho_0{try_double(tbl, "initial_density")},
alpha{try_double(tbl, "alpha")},
A{try_double(tbl, "interfacial_tension_control")},
nu{try_double(tbl, "kinematic_viscosity")},
mu{nu*rho_0},
beta{try_double(tbl, "interface_thickness_control")},
cs2{init_cs2(alpha)},
ics2{1.0/cs2},
rlx{init_rlx_param(nu, cs2)}
{
  phi    = torch::zeros({9}, torch::kCUDA);
  rho    = torch::zeros({rows,cols,1}, torch::kCUDA);
  p      = torch::zeros({rows,cols,1}, torch::kCUDA);
  eta    = torch::zeros({rows,cols,9}, torch::kCUDA);
  C      = torch::zeros({rows,cols,9}, torch::kCUDA);
  omega1 = torch::zeros({rows,cols,9}, torch::kCUDA);
  omega2 = torch::zeros({rows,cols,9}, torch::kCUDA);
  omega3 = torch::zeros({rows,cols,9}, torch::kCUDA);
  equ_f  = torch::zeros({rows,cols,9}, torch::kCUDA);
  col_f  = torch::zeros({rows,cols,9}, torch::kCUDA);
  adv_f  = torch::zeros({rows,cols,9}, torch::kCUDA);
  init_phi(phi, alpha);
  init_eta(eta, cs2, E);
}

double colour::init_cs2(double alpha){ return 3.0*(1.0-alpha)/5.0; }
double colour::init_rlx_param(double nu, double cs2)
{ return 1.0/( 0.5 + nu/cs2 ); }

double colour::try_double(const toml::node_view<toml::node>& tbl, const std::string name)
{
  std::optional<double> op = tbl[name].value<double>();
  if (op.has_value()) return op.value();
  else throw std::runtime_error(name + "not defined in parameters file");
  return 0.0;
}

void colour::init_eta(torch::Tensor& eta_, double cs2, const torch::Tensor& E)
{
  eta_.copy_(
    1.0 + 0.5*(3.0*cs2 - 1.0)*(3.0*E.mul(E).sum(0) - 4.0)
  );
}

void colour::init_phi(torch::Tensor& phi_, double alpha)
{
  double a = 0.2*(1.0 - alpha);
  double b = 0.05*(1.0 - alpha);
  torch::Tensor temp = torch::tensor(
    {alpha, a, a, a, a, b, b, b, b},
    torch::kCUDA);
  phi.copy_(temp);
}

std::ostream& operator<<(std::ostream& os, const colour& c)
{
  return os << " FLUID parameters:\n"
  << "rho_0=" << c.rho_0
  << "\nalpha=" << c.alpha
  << "\nnu=" << c.nu
  << "\nmu=" << c.mu
  << "\nA=" << c.A
  << "\nbeta=" << c.beta
  << "\ncs2=" << c.cs2
  << "\nics2=" << c.ics2
  << "\nrlx=" << c.rlx
  << std::endl;
}

