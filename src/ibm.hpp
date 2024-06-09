#ifndef IBM_HPP
#define IBM_HPP

#include <torch/torch.h>
#include <vector>

#include "toml++/toml.hpp"

struct marker
{
  torch::Tensor r;
  torch::indexing::Slice rows, cols;
  torch::Tensor phi;

  marker(double x, double y);
  void set_box(double x, double y);
  torch::Tensor calc_phi(const torch::Tensor& r);
  double calc_phi(double r);
};

class ibm
{
public:
  ibm(const toml::table& tbl, const std::string& name,
      const torch::Device& dev, int m_max=5);
  torch::Tensor eulerian_force_density(const torch::Tensor& u_0, const torch::Tensor& rho_0);
  const torch::indexing::Slice rows, cols;
private:
  const int m_max;
  std::vector<marker> markers; // List of markers defining the immersed boundary
  torch::Tensor u, rho, F, uj, fj;

  torch::indexing::Slice get_roi(const toml::table& tbl, const std::string& name, char type);
};

#endif
