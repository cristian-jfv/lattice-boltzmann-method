#include <iostream>
#include <cmath>
#include <torch/torch.h>

using std::cerr;

const torch::Tensor W = torch::tensor(
  {4.0/ 9.0,
    1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

const torch::Tensor E = torch::tensor(
  {{0.0, 1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
   {0.0, 0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0}},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

const torch::Tensor E_E = (E*E).sum_to_size({1,9}).clone().detach();

void equilibrium
(
  torch::Tensor &f_equ,
  const torch::Tensor &u,
  const torch::Tensor &rho,
  const torch::Tensor &phi,
  const torch::Tensor &equ_factor,
  const double cs
);

class color
{
public:
  torch::Tensor rho;
  torch::Tensor f;
  torch::Tensor f_equ;
  const torch::Tensor equ_factor;
  const double alpha;
  const double cs2;

  color(int L, double R, double rho_0, double alpha, bool invert_sigmoid=false):
  equ_factor{init_equ_factor(alpha)},
  alpha{alpha},
  cs2{init_cs2(alpha)},
  phi{init_phi(alpha)}
  {
    rho = torch::zeros({L, L, 1}, torch::kCUDA);
    init_rho(rho, rho_0, L, R, invert_sigmoid);
    f = torch::zeros({L, L, 9}, torch::kCUDA);
    f_equ = torch::zeros_like(f);
  }

  void equilibrium
  (
    const torch::Tensor &u
  )
  {
    torch::Tensor u_u = (u*u).sum_to_size(rho.sizes());
    torch::Tensor E_u = torch::matmul(u, E);
    f_equ = (rho*(phi + torch::mul(3.0*E_u*equ_factor + 4.5*E_u.pow(2) - 1.5*u_u, W))).clone().detach();
  }

private:
  const torch::Tensor phi;

  torch::Tensor init_equ_factor(double alpha)
  {
    const double cs2 = init_cs2(alpha);
    return (1.0 + 0.5*(3.0*cs2 - 1.0)*(3.0*E_E-4.0)).clone().detach();
  }

  torch::Tensor init_phi(double alpha)
  {
    const double a = 0.2*(1-alpha);
    const double b = 0.05*(1-alpha);
    return torch::tensor({alpha, a, a, a, a, b, b, b, b}, torch::kCUDA);
  }

  void init_rho(torch::Tensor &rho, double rho_0, double L, double R, bool invert)
  {
    int rows = rho.size(0);
    int cols = rho.size(1);
    double C = L/2.0;

    for(int r=0; r<rows; r++)
    {
      for(int c=0; c<cols; c++)
      {
        double s = std::sqrt((r-C)*(r-C) + (c-C)*(c-C));
        double ans = 0.0;
        if(invert) ans = 1.0 - sigmoid(2.0*(s-R));
        else ans = sigmoid(2.0*(s-R));
        rho[r][c] = rho_0*ans;
      }
    }
  }

  double sigmoid(double x) { return 1.0/(1.0 + std::exp(-x)); }

  double init_cs2(double alpha) { return 0.6*(1-alpha); }
};

int main(int argc, char *argv[])
{
  using torch::Tensor;

  torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(torch::kDouble));
  if (!torch::cuda::is_available())
  {
    cerr << "CUDA is NOT available\n";
  }
  const torch::Device dev = torch::kCUDA;

  // Parameters
  const int L = 100; // Domain size
  const double R = 25.0; // Droplet radius
  const double r_rho_0 = 10.0;
  const double b_rho_0 = 1.0;

  // Initialization
  color r{L, R, r_rho_0, 0.92, true};
  color b{L, R, b_rho_0, 0.2};
  Tensor u = torch::zeros({L, L, 2}, dev);

  // Main loop
  for (int t=0; t < 1000; t++)
  {
    if (false)
    {
      // Save results
    }

    // Compute equilibrium distributions
    r.equilibrium(u);
    b.equilibrium(u);

    // Compute BGK operator
    // Compute perturbation operator
    // Compute redistribution operator
    // Collision
    // Advection
    // Boundary conditions
    // Compute macroscopics quantities

  }

  return 0;
}


