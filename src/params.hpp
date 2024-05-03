#ifndef PARAMS_HPP
#define PARAMS_HPP
#include <iostream>
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
  double Re;
  double nu;
  int l;
  double dx;
  double dt;
  double u;
  int Y,X;
  lattice(const toml::table& tbl, const flow &fp);
};

std::ostream& operator<<(std::ostream& os, const lattice& p);
}
#endif
