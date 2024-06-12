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

const torch::Tensor M_original = torch::tensor(
  {{ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0},
   {-4.0, -1.0, -1.0, -1.0, -1.0,  2.0,  2.0,  2.0,  2.0},
   { 4.0, -2.0, -2.0, -2.0, -2.0,  1.0,  1.0,  1.0,  1.0},
   { 0.0,  1.0,  0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
   { 0.0, -2.0,  0.0,  2.0,  0.0,  1.0, -1.0, -1.0,  1.0},
   { 0.0,  0.0,  1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0},
   { 0.0,  0.0, -2.0,  0.0,  2.0,  1.0,  1.0, -1.0, -1.0},
   { 0.0,  1.0, -1.0,  1.0, -1.0,  0.0,  0.0,  0.0,  0.0},
   { 0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  1.0, -1.0}},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));
// Use the tranpose to enable matrix multiplication latter on
const torch::Tensor M = M_original.t().clone().detach();
const torch::Tensor M_inv = M_original.inverse().t().clone().detach();

void phase_field
(
  torch::Tensor &rho_n,
  const torch::Tensor &r_rho,
  double r_rho_0,
  const torch::Tensor &b_rho,
  double b_rho_0
);

struct relaxation_function
{
private:
  const double delta;
  const double r_s_nu;
  const double b_s_nu;
  const double beta, gamma, epsilon, eta, xi;

  double init_s_nu(double nu, double cs2)
  { return 1.0/( 0.5 + nu/cs2 ); }

  double init_beta(double r_s_nu, double b_s_nu)
  { return 2.0*r_s_nu*b_s_nu/(r_s_nu + b_s_nu); }

  double init_gamma(double r_s_nu, double beta, double delta)
  { return 2.0*(r_s_nu - beta)/delta; }

  double init_epsilon(double gamma, double delta)
  { return -gamma/(2.0*delta); }

  double init_eta(double b_s_nu, double beta, double delta)
  { return 2.0*(beta - b_s_nu)/delta; }

  double init_xi(double eta, double delta)
  { return eta/(2.0*delta); }

  double eval(double psi) // Scalar function to evaluate the relaxation parameter
  {
    if (psi > delta) return r_s_nu;
    else if (delta >= psi && psi > 0) return beta + gamma*psi + epsilon*psi*psi;
    else if (0 >= psi && psi >= -delta) return beta + eta*psi + xi*psi*psi;
    return b_s_nu;
  }

public:
  relaxation_function(double r_nu, double r_cs2, double b_nu, double b_cs2, double delta):
  delta{delta},
  r_s_nu{init_s_nu(r_nu, r_cs2)},
  b_s_nu{init_s_nu(b_nu, b_cs2)},
  beta{init_beta(r_s_nu, b_s_nu)},
  gamma{init_gamma(r_s_nu, beta, delta)},
  epsilon{init_epsilon(gamma, delta)},
  eta{init_eta(b_s_nu, beta, delta)},
  xi{init_xi(eta, delta)}
  {
    // k_nu: kinematic viscosity
    // k_cs2: squared numeric sound speed
  }

  void eval(torch::Tensor &s_nu, const torch::Tensor &psi)
  {
    auto mask = (psi > delta).to(torch::kDouble);
    auto elements = mask*r_s_nu;
    s_nu.masked_fill_(mask.to(torch::kBool), elements);

    mask.copy_( (delta >= psi) * (psi > 0) ).to(torch::kDouble);
    elements.copy_(mask*(beta + gamma*psi + epsilon*psi*psi));
    s_nu.masked_fill_(mask.to(torch::kBool), elements);

    mask.copy_( (0 >= psi) * (psi >= -delta) ).to(torch::kDouble);
    elements.copy_(mask*(beta + eta*psi + xi*psi*psi));
    s_nu.masked_fill_(mask.to(torch::kBool), elements);

    mask.copy_( psi < -delta ).to(torch::kDouble);
    elements.copy_(mask*b_s_nu);
    s_nu.masked_fill_(mask.to(torch::kBool), elements);
  }

};

class color
{
public:
  torch::Tensor rho;
  torch::Tensor f;
  torch::Tensor f_equ;
  const double alpha;
  const double cs2;
  const double nu;

  color(int L, double R, double rho_0, double alpha, double nu, bool invert_sigmoid=false):
  alpha{alpha},
  cs2{init_cs2(alpha)},
  nu{nu}
  {
    rho = torch::zeros({L, L, 1}, torch::kCUDA);
    init_rho(rho, rho_0, L, R, invert_sigmoid);
    f = torch::zeros({L, L, 9}, torch::kCUDA);
    f_equ = torch::zeros_like(f);
  }

private:

  void mrt_operator(torch::Tensor &f, const torch::Tensor &rho, const torch::Tensor &u)
  {
    
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

  // Initialization
  color r{L, R, /*rho_0=*/10.0, /*alpha=*/0.92, /*nu=*/0.1667, true};
  color b{L, R, /*rho_0=*/ 1.0, /*alpha=*/0.2,  /*nu=*/0.1667};
  relaxation_function s_nu{r.nu, r.cs2, b.nu, b.cs2, /*delta=*/0.1};
  Tensor u = torch::zeros({L, L, 2}, dev);

  // Main loop
  for (int t=0; t < 1000; t++)
  {
    if (false)
    {
      // Save results
    }

    // Compute equilibrium distributions


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

void eval_phase_field
(
  torch::Tensor &rho_n,
  const torch::Tensor &r_rho,
  double r_rho_0,
  const torch::Tensor &b_rho,
  double b_rho_0
)
{
  rho_n = ((r_rho/r_rho_0 - b_rho/b_rho_0)
          /
          (r_rho/r_rho_0 + b_rho/b_rho_0)).clone().detach();
}
