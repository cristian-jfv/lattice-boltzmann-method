#ifndef PARAMS_HPP
#define PARAMS_HPP
#include <iostream>
#include <string>
#include <toml++/toml.hpp>

namespace params
{
struct flow
{
  double nu{0};
  double u{0};
  double l{0};
  double rho_0{0};
  double Re{0};
  flow(const toml::table& tbl);
};

std::ostream& operator<<(std::ostream& os, const flow& p);

struct lattice
{
  const double cs2{1.0/3.0};
  double tau;
  double omega;
  double Re;
  double nu;
  int l;
  double dx;
  double dt;
  int T;
  double u;
  int Y,X;
  lattice(const toml::table& tbl, const flow& fp);
};

std::ostream& operator<<(std::ostream& os, const lattice& p);

struct simulation
{
  double stop_time;
  double snapshot_period;
  int total_steps;
  int snapshot_steps;
  int total_snapshots;
  std::string file_prefix;
  simulation(const toml::table& tbl, const lattice& lp);
  bool snapshot(int step) const;
};

std::ostream& operator<<(std::ostream& os, const simulation& p);
}
#endif
