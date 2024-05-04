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

  // Set the characteristic length to the nearest odd integer
  if ((int)std::ceil(fp.l/dx) % 2 != 0) l = std::ceil(fp.l/dx);
  else l = std::floor(fp.l/dx);

  omega = 1.0/tau;
  Re = fp.Re;
  nu = cs2*(tau - 0.5);
  u = fp.Re*nu/l;
  dt = cs2*(tau - 0.5)*(dx*dx)/fp.nu;
  T = std::ceil(1.0/dt);
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
   << "omega=" << p.omega << "\n"
   << "dx=" << p.dx << " m\n"
   << "l=" << p.l << "\n"
   << "nu=" << p.nu << "\n"
   << "u=" << p.u << "\n"
   << "dt=" << p.dt << "\n"
   << "T=" << p.T << "\n"
   << "X=" << p.X << "\n"
   << "Y=" << p.Y << std::endl;
}


params::simulation::simulation(const toml::table& tbl, const lattice& lp)
{
  using std::optional;

  optional<double> op_stop = tbl["simulation"]["stop_time"].value<double>();
  if (op_stop.has_value()) stop_time = op_stop.value();
  else throw std::runtime_error("stop_time not defined in parameters file");

  optional<double> op_period = tbl["simulation"]["snapshot_period"].value<double>();
  if (op_period.has_value()) snapshot_period = op_period.value();
  else throw std::runtime_error("snapshot_period not defined in parameters file");

  optional<std::string> op_prefix = tbl["simulation"]["file_prefix"].value<std::string>();
  if (op_prefix.has_value()) file_prefix = op_prefix.value();
  else throw std::runtime_error("file_prefix not defined in parameters file");

  total_steps = std::ceil(stop_time*lp.T);
  snapshot_steps = std::ceil(snapshot_period*lp.T);
  total_snapshots = std::ceil((total_steps+0.0)/snapshot_steps);
}

bool params::simulation::snapshot(int step) const
{
  if (step % snapshot_steps == 0) return true;
  return false;
}

std::ostream& params::operator<<(std::ostream& os, const params::simulation& p)
{
  return os << "Simulation parameters:\n"
  << "stop time: " << p.stop_time << " s (" << p.total_steps << " steps)\n"
  << "saving results each " << p.snapshot_period << " s (" << p.snapshot_steps << " steps)\n"
  << "for a total of " << p.total_snapshots << " snapshots" << std::endl;
}
