#include "params.hpp"
#include <cmath>
#include <optional>
#include <ostream>
#include <stdexcept>

params::flow::flow(const toml::table& tbl)
{
  using std::optional;

  optional<double> op_rho = tbl["flow"]["initial_density"].value<double>();
  if (op_rho.has_value()) rho_0=op_rho.value();
  else throw std::runtime_error("initial_density not defined in parameters file");


  optional<double> op_nu = tbl["flow"]["kinematic_viscosity"].value<double>();
  if (op_nu.has_value()) nu=op_nu.value();
  else throw std::runtime_error("kinematic_viscosity not defined in parameters file");

  optional<double> op_u = tbl["flow"]["characteristic_velocity"].value<double>();
  if (op_u.has_value()) u=op_u.value();
  else throw std::runtime_error("characteristic_velocity not defined in parameters file");

  optional<double> op_l = tbl["flow"]["characteristic_length"].value<double>();
  if (op_l.has_value()) l=op_l.value();
  else throw std::runtime_error("characteristic_length not defined in parameters file");

  Re = u*l/nu;
}


params::lattice::lattice(const toml::table& tbl, const params::flow &fp):
cs2{1.0/3.0}
{
  using std::optional;
  optional<double> op_tau = tbl["lattice"]["relaxation_time"].value<double>();
  if (op_tau.has_value()) tau = op_tau.value();
  else throw std::runtime_error("relaxation_time not defined in parameters file");

  optional<double> op_dx = tbl["lattice"]["lattice_spacing"].value<double>();
  if (op_dx.has_value()) dx = op_dx.value();
  else throw std::runtime_error("lattice_spacing not defined in parameters file");

  optional<double> op_x_m = tbl["lattice"]["x_multiplier"].value<double>();
  if (!op_x_m.has_value())
    throw std::runtime_error("x_multiplier not defined in parameters file");

  optional<double> op_y_m = tbl["lattice"]["y_multiplier"].value<double>();
  if (!op_y_m.has_value())
    throw std::runtime_error("y_multiplier not defined in parameters file");

  const double x_mult{op_x_m.value()};
  const double y_mult{op_y_m.value()};
  l = std::ceil(fp.l/dx); // characteristic length in lattice units

  Re = fp.Re;
  nu = cs2*(tau - 0.5);
  u = fp.Re*nu/l;
  dt = cs2*(tau - 0.5)*(dx*dx)/fp.nu;
  X = std::ceil(l*x_mult);
  Y = std::ceil(l*y_mult);
}

std::ostream& params::operator<<(std::ostream& os, const params::flow& p)
{
  return os << "Flow parameters:\n"
   << "nu=" << p.nu << " m2/s\n"
   << "u=" << p.u << " m/s\n"
   << "l=" << p.l << " m\n"
   << "rho_0=" << p.rho_0 << " kg/m3\n"
   << "Re=" << p.Re << std::endl;
}

std::ostream& params::operator<<(std::ostream& os, const params::lattice& p)
{
  return os << "Lattice parameters:" << std::endl
   << "Re=" << p.Re << "\n"
   << "tau=" << p.tau << "\n"
   << "dx=" << p.dx << " m\n"
   << "l=" << p.l << "\n"
   << "nu=" << p.nu << "\n"
   << "u=" << p.u << "\n"
   << "dt=" << p.dt << "\n"
   << "T=" << 1.0/p.dt << "\n"
   << "X=" << p.X << "\n"
   << "Y=" << p.Y << std::endl;
}
