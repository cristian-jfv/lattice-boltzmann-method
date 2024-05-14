#include "ibm.hpp"
#include "utils.hpp"
#include <ATen/TensorIndexing.h>
#include <ATen/ops/zeros_like.h>
#include <c10/core/DeviceType.h>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <torch/csrc/autograd/generated/variable_factories.h>

const torch::Tensor stencil = torch::tensor({
       {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
       {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}}, torch::kCUDA);

marker::marker(double x, double y)
{
  set_box(x, y);
}

void marker::set_box(double x, double y)
{
  using torch::indexing::Ellipsis;
  using torch::indexing::Slice;

  r = torch::tensor({{x}, {y}}, torch::kCUDA);
  torch::Tensor s = r - ( stencil + torch::floor(r) - 1.0);
  //utils::print("s",s);
  phi = calc_phi(s);

  long start = (long)std::floor(x) - 1;
  long stop = start + 4;
  rows = Slice(start, stop);

  start = (long)std::floor(y) - 1;
  stop = start + 4;
  cols = Slice(start, stop);
}

double marker::calc_phi(double _r)
{
  double r = std::abs(_r);
  if (r <= 1) return 0.125*(3.0 - 2.0*r + std::sqrt(1.0 + 4.0*r - 4.0*r*r));
  else if (r <= 2) return 0.125*(5.0 - 2.0*r - std::sqrt(-7.0 + 12.0*r - 4.0*r*r));
  return 0.0;
}

torch::Tensor marker::calc_phi(const torch::Tensor& s)
{
  using torch::indexing::Slice;
  torch::Tensor a = torch::zeros_like(s);
  auto sizes = a.sizes();
  for (int r = 0; r < sizes[0]; r++)
    for (int c = 0; c < sizes[1]; c++)
      a[r][c] = calc_phi(s[r][c].item<double>());

  return a.index({0,Slice()})*a.index({1,Slice()});
}

ibm::ibm
(
  const toml::table& tbl,
  const std::string& name,
  const torch::Device& dev,
  int m_max
):
rows{get_roi(tbl, name, 'r')},
cols{get_roi(tbl, name, 'c')},
m_max{m_max}
{
  // TODO: Consider the case of multiple boundaries
  // int i = 0;
  // while (!!tbl["boundaries"][i])

  int64_t r_off = rows.start().maybe_as_int().value();
  int64_t c_off = cols.start().maybe_as_int().value();

  // Read boundary at the toml file
  const toml::array* x = tbl[name]["x"].as_array();
  const toml::array* y = tbl[name]["y"].as_array();

  // Build markersligth
  const size_t size = x->size();
  long r_min{1000000}, r_max{0};
  long c_min{1000000}, c_max{0};
  for (size_t i = 0; i < size; i++)
  {
    double x_m, y_m;
    std::optional<double> x_o = x->at(i).value<double>();
    if (x_o.has_value()) x_m = x_o.value();
    else throw std::runtime_error("Cannot parse x coordinate");
    std::optional<double> y_o = y->at(i).value<double>();
    if (y_o.has_value()) y_m = y_o.value();
    else throw std::runtime_error("Cannot parse y coordinate");

    // Track region of interest (ROI)
    if (r_min > (int)(std::floor(x_m) - 1)) r_min = (int)(std::floor(x_m) - 1);
    if (r_max < (int)(std::floor(x_m) + 2)) r_max = (int)(std::floor(x_m) + 2);
    if (c_min > (int)(std::floor(y_m) - 1)) c_min = (int)(std::floor(y_m) - 1);
    if (c_max < (int)(std::floor(y_m) + 2)) c_max = (int)(std::floor(y_m) + 2);

    markers.push_back(marker(x_m - r_off , y_m - c_off));
  }
  utils::print("Immersed boundary data:");
  // Give size to the internal copies of velocity, density, and Eulerian forces
  u = torch::zeros({r_max-r_min + 1, c_max-c_min + 1, 2}, dev);
  utils::print("u.sizes", u.sizes());
  rho = torch::zeros({r_max-r_min + 1, c_max-c_min + 1, 1}, dev);
  utils::print("rho.sizes", rho.sizes());
  F = torch::zeros({r_max-r_min + 1, c_max-c_min + 1, 2, m_max}, dev);
  utils::print("F.sizes", F.sizes());
  uj = torch::zeros({2, (long)size}, dev);
  utils::print("uj.sizes", uj.sizes());
  fj = torch::zeros_like(uj, dev);
  utils::print("fj.sizes", fj.sizes());

  // Statistics
  utils::print("markers.size", markers.size());
  utils::print("rows", rows);
  utils::print("cols", cols);
}

torch::indexing::Slice ibm::get_roi
(
  const toml::table& tbl, const std::string& name, char type
)
{
  using torch::indexing::Slice;
  // Read boundary at the toml file
  const toml::array* x = tbl[name]["x"].as_array();
  const toml::array* y = tbl[name]["y"].as_array();

  // Build markers
  size_t size = x->size();
  long r_min{1000000}, r_max{0};
  long c_min{1000000}, c_max{0};
  for (size_t i = 0; i < size; i++)
  {
    double x_m, y_m;
    std::optional<double> x_o = x->at(i).value<double>();
    if (x_o.has_value()) x_m = x_o.value();
    else throw std::runtime_error("Cannot parse x coordinate");
    std::optional<double> y_o = y->at(i).value<double>();
    if (y_o.has_value()) y_m = y_o.value();
    else throw std::runtime_error("Cannot parse y coordinate");

    // Track region of interest (ROI)
    if (r_min > (int)(std::floor(x_m) - 1)) r_min = (int)(std::floor(x_m) - 1);
    if (r_max < (int)(std::floor(x_m) + 2)) r_max = (int)(std::floor(x_m) + 2);
    if (c_min > (int)(std::floor(y_m) - 1)) c_min = (int)(std::floor(y_m) - 1);
    if (c_max < (int)(std::floor(y_m) + 2)) c_max = (int)(std::floor(y_m) + 2);
  }
  if (type=='r') return Slice(r_min, r_max + 1);
  else if (type=='c') return Slice(c_min, c_max + 1);

  return Slice(r_min, r_max + 1);
}

torch::Tensor ibm::eulerian_force_density(const torch::Tensor& u_0, const torch::Tensor& rho_0)
{
  using torch::indexing::Ellipsis;
  using torch::indexing::Slice;
  // Copy uncorrected density and velocity
  u = u_0.index({rows, cols, Ellipsis}).clone().detach();
  rho = rho_0.index({rows, cols, Ellipsis}).clone().detach();
  F = torch::zeros_like(F); // make sure F starts as zeros to easily add results over
  for (int n=1; n<m_max; n++)
  {
    long i = 0;
    for (const auto& m : markers)
    {
      auto box = u.index({m.rows, m.cols, Ellipsis}).clone().detach().reshape({16,2});
      // Interpolate u(x) at r_j
      uj.index({Slice(), i}) = m.phi.matmul(box).clone().detach();
      //utils::print("phi", m.phi);
      auto rhoj = m.phi.matmul(rho.index({m.rows, m.cols, Ellipsis}).reshape({16,1}));
      //utils::print("rhoj", rhoj);
      fj.index({Slice(), i}) = -2.0*rhoj*uj.index({Slice(), i}).clone().detach();

      // This implements deltaF for each marker
      F.index({m.rows, m.cols, Slice(), n}) +=
        (m.phi.reshape({4, 4, 1})*fj.index({Slice(), i}).unsqueeze(1).t())
        .clone().detach();
      i++;
    }

    u += 0.5*F.index({Ellipsis, n})/rho;
  }

  return torch::sum(F, 3); //.sum_to_size(u.sizes());
}

