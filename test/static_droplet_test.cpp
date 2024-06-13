#include <ATen/TensorIndexing.h>
#include <ATen/ops/diagflat.h>
#include <c10/core/DeviceType.h>
#include <iostream>
#include <cmath>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

using std::cerr;
using std::cout;
using std::endl;
using torch::Tensor;
using torch::indexing::Ellipsis;
using torch::indexing::Slice;

const Tensor W = torch::tensor(
  {4.0/ 9.0,
    1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

const Tensor E = torch::tensor(
  {{0.0, 1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
   {0.0, 0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0}},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

const Tensor E_E = (E*E).sum_to_size({1,9}).clone().detach();

const Tensor M_original = torch::tensor(
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
const Tensor M = M_original.t().clone().detach();
const Tensor Mi = M_original.inverse().t().clone().detach();

void eval_phase_field
(
  torch::Tensor &rho_n,
  const torch::Tensor &r_rho,
  double r_rho_0,
  const torch::Tensor &b_rho,
  double b_rho_0
);

void eval_local_curvature(Tensor &K, const Tensor &n);

class partial_derivatives
{
private:
// Kernels for partial derivatives
const Tensor kernel_partial_x = 3.0*torch::tensor(
  {{-1.0/36.0, 0.0, 1.0/36.0},
   { -1.0/9.0, 0.0,  1.0/9.0},
   {-1.0/36.0, 0.0, 1.0/36.0}},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

const Tensor kernel_partial_y = 3.0*torch::tensor(
  {{1.0/36.0, 1.0/9.0, 1.0/36.0},
   {     0.0,     0.0,      0.0},
   {-1.0/36.0, -1.0/9.0, -1.0/36.0}},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

torch::nn::Conv2d initialize_convolution(const Tensor &kernel)
{
  auto conv_options = torch::nn::Conv2dOptions(
    /*in_channels=*/1, /*out_channels=*/1, /*kernel_size=*/3)
                               .padding({1,1})
                               .padding_mode(torch::kReplicate)
                               .bias(false);

  torch::nn::Conv2d ans(conv_options);
  ans->weight = kernel.reshape({1, 1, 3, 3}).clone().detach();
  ans->bias = torch::tensor({0.0},torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));
  return ans;
}

// Convolution operators for partial derivatives
torch::nn::Conv2d partial_x = nullptr;
torch::nn::Conv2d partial_y = nullptr;

public:
  partial_derivatives()
  {
    partial_x = initialize_convolution(kernel_partial_x);
    partial_y = initialize_convolution(kernel_partial_y);
  }

  Tensor x(const Tensor &psi)
  {
    return partial_x->forward(psi.unsqueeze(0).unsqueeze(0).squeeze(-1))
    .squeeze(0).squeeze(0).clone().detach();
  }

  Tensor y(const Tensor &psi)
  {
    return partial_y->forward(psi.unsqueeze(0).unsqueeze(0).squeeze(-1))
    .squeeze(0).squeeze(0).clone().detach();
  }

  void grad(Tensor &ans, const Tensor &psi)
  {
    ans.index({Ellipsis,0}) = x(psi);
    ans.index({Ellipsis,1}) = y(psi);
  }

};
partial_derivatives partial{};

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
  Tensor rho;
  Tensor f;
  Tensor m_eq;
  Tensor C;
  Tensor S;
  const double alpha;
  const double cs2;
  const double nu;
  const double rho_0;
  const double omega;
  const double A;
  const double s_e = 1.25;
  const double s_zeta = 1.14;
  const double s_q = 1.6;

  color(int L, double R, double rho_0, double alpha, double nu, double A, bool invert_sigmoid=false):
  alpha{alpha},
  cs2{init_cs2(alpha)},
  nu{nu},
  rho_0{rho_0},
  omega{init_omega(nu, cs2)},
  A{A}
  {
    rho = torch::zeros({L, L, 1}, torch::kCUDA);
    init_rho(rho, rho_0, L, R, invert_sigmoid);
    f = torch::zeros({L, L, 9}, torch::kCUDA);
    m_eq = torch::zeros_like(f);
    C = torch::zeros_like(f);
    Tensor temp_s = torch::diagflat(torch::tensor({{0.0, s_e, s_zeta, 0.0, s_q, 0.0, s_q, 0.0, 0.0}},
             torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA)));
    S = temp_s.unsqueeze(0).unsqueeze(0).repeat({L, L, 1, 1});
  }

private:

  void mrtp_operator(Tensor &Omega, const Tensor &u, const Tensor &F, const Tensor &s_nu)
  {
    // [(fM - m^eq + F)S + C]M^-1   (all matrices are tranposed)
    update_m_eq(u);
    update_C(u, s_nu);
    update_S(s_nu);
    Omega.copy_(
      (S.matmul(( f.matmul(M) - m_eq + A*(1.0 - 0.5*omega)*F ).unsqueeze(-1)).squeeze(-1) + C).matmul(Mi)
    );
  }

  void update_S(const Tensor &s_nu)
  {
    S.index({Ellipsis, 7, 7}) = s_nu.clone().detach();
    S.index({Ellipsis, 8, 8}) = s_nu.clone().detach();
  }

  void update_C(const Tensor &u, const Tensor &s_nu)
  {
    #define ux u.index({Ellipsis,0})
    #define uy u.index({Ellipsis,1})
    const double a = 1.8*alpha - 0.8;
    Tensor Qx = a*rho*ux;
    Tensor Qy = a*rho*uy;

    C.index({Ellipsis,1}) = 3.0*(1.0 - 0.5*s_e)*(partial.x(Qx) + partial.y(Qy)).clone().detach();
    C.index({Ellipsis,7}) = (1.0 - 0.5*s_nu)*(partial.x(Qx) - partial.y(Qy)).clone().detach();

  }

  void update_m_eq(const Tensor &u)
  {
    #define ux u.index({Ellipsis,0})
    #define uy u.index({Ellipsis,1})
    Tensor u_u = (u*u).sum_to_size(rho.sizes());
    m_eq.index({Ellipsis,0}) = 1.0;
    m_eq.index({Ellipsis,1}) = (-3.6*alpha - 0.4 + 3.0*u_u).clone().detach();
    m_eq.index({Ellipsis,2}) = (5.4*alpha - 1.4 - 3.0*u_u).clone().detach();
    m_eq.index({Ellipsis,3}) = ux.clone().detach();
    m_eq.index({Ellipsis,4}) = (-1.8*alpha - 0.2)*ux.clone().detach();
    m_eq.index({Ellipsis,5}) = uy.clone().detach();
    m_eq.index({Ellipsis,6}) = (-1.8*alpha - 0.2)*uy.clone().detach();
    m_eq.index({Ellipsis,7}) = (ux*ux - uy*uy).clone().detach();
    m_eq.index({Ellipsis,8}) = (ux*uy).clone().detach();
  }

  double init_omega(double nu, double cs2) { return 0.5 + nu/cs2; }

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


void build_F(Tensor &F, const Tensor &u, const Tensor &Fs);

int main(int argc, char *argv[])
{
  torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(torch::kDouble));
  if (!torch::cuda::is_available())
  {
    cerr << "CUDA is NOT available\n";
  }
  const torch::Device dev = torch::kCUDA;

  // Parameters
  const int L = 5; // Domain size
  const double R = 25.0; // Droplet radius

  const double sigma = 0.1; // Interfacial tension coefficient

  // Initialization
  color r{L, R, /*rho_0=*/10.0, /*alpha=*/0.92, /*nu=*/0.1667, /*A=*/0.5, true};
  color b{L, R, /*rho_0=*/ 1.0, /*alpha=*/0.2,  /*nu=*/0.1667, /*A=*/0.5,};
  relaxation_function s_nu_function{r.nu, r.cs2, b.nu, b.cs2, /*delta=*/0.1};
  Tensor u = torch::zeros({L, L, 2}, dev);
  Tensor rho = torch::zeros({L, L, 1}, dev);
  Tensor s_nu = torch::zeros({L, L, 1}, dev);
  Tensor rho_n = torch::zeros({L, L, 1}, dev);
  Tensor grad_rho_n = torch::zeros({L, L, 2}, dev);
  Tensor n = torch::zeros({L,L,2}, dev);
  Tensor K = torch::zeros({L,L,1}, dev);
  Tensor F_s = torch::zeros({L,L,2}, dev);
  Tensor F = torch::zeros({L,L,9}, dev);


  // Main loop
  cout << "main loop start" << endl;
  cout << torch::tensor({0.0}) << endl;
  for (int t=0; t < 1000; t++)
  {
    if (false)
    {
      // Save results
    }

    // Compute equilibrium distributions
    eval_phase_field(rho_n, r.rho, r.rho_0, b.rho, b.rho_0);
    partial.grad(grad_rho_n, rho_n);
    n = grad_rho_n/torch::norm(rho_n, 2, -1).unsqueeze(-1);
    eval_local_curvature(K, n);
    F_s = -0.5*sigma*K*grad_rho_n;
    build_F(F, u, F_s);
    s_nu_function.eval(s_nu, rho_n);

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

void build_F(Tensor &F, const Tensor &u, const Tensor &Fs)
{
  #define ux u.index({Ellipsis,0})
  #define uy u.index({Ellipsis,1})
  #define Fsx Fs.index({Ellipsis,0})
  #define Fsy Fs.index({Ellipsis,1})

  F.index({Ellipsis,0}) = 0.0;
  F.index({Ellipsis,1}) = 6.0*(ux*Fsx + uy*Fsy).clone().detach();
  F.index({Ellipsis,2}) = -1.0*F.index({Ellipsis,1}).clone().detach();
  F.index({Ellipsis,3}) = Fsx.clone().detach();
  F.index({Ellipsis,4}) = -1.0*Fsx.clone().detach();
  F.index({Ellipsis,5}) = Fsy.clone().detach();
  F.index({Ellipsis,6}) = -1.0*Fsy.clone().detach();
  F.index({Ellipsis,7}) = 2.0*(ux*Fsx - uy*Fsy).clone().detach();
  F.index({Ellipsis,8}) = (ux*Fsy + uy*Fsx).clone().detach();
}

void eval_local_curvature(Tensor &K, const Tensor &n)
{
  #define nx n.index({Ellipsis,0})
  #define ny n.index({Ellipsis,1})
  K = (nx*ny*(partial.y(nx) + partial.x(ny))
      - nx.pow(2.0)*partial.y(ny) - ny.pow(2.0)*partial.x(nx))
    .unsqueeze(-1).clone().detach();
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
