#ifndef COLOUR_HPP
#define COLOUR_HPP

#include "toml++/impl/node_view.hpp"
#include <string>
#include <torch/torch.h>
#include <toml++/toml.hpp>

class colour
{
  public:
  const double rho_0;
  const double alpha;
  const double A;
  const double nu; // kinematic viscocity
  const double mu; // dynamic viscosity
  const double beta;
  const double cs2; // speed of sound
  const double ics2; // 1/speed of sound

  torch::Tensor rho;
  torch::Tensor p;
  torch::Tensor phi;
  torch::Tensor C;
  torch::Tensor eta;
  torch::Tensor omega1;
  torch::Tensor omega2;
  torch::Tensor omega3;
  torch::Tensor equ_f;
  torch::Tensor col_f;
  torch::Tensor adv_f;

  colour(const toml::node_view<toml::node>& tbl, int R, int C, const torch::Tensor& E);

  private:
  double try_double(const toml::node_view<toml::node>& tbl, const std::string name);
  double init_cs2(double alpha);
  void init_eta(torch::Tensor& eta_, double cs2, const torch::Tensor& E);
  void init_phi(torch::Tensor& phi_, double alpha);
};

std::ostream& operator<<(std::ostream& os, const colour& p);

#endif
