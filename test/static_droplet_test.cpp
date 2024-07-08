#include <ATen/TensorIndexing.h>
#include <c10/core/DeviceType.h>
#include <iostream>
#include <cmath>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/functional/normalization.h>
#include <torch/nn/options/normalization.h>
#include <torch/torch.h>

#include "../src/solver.hpp"
#include "../src/utils.hpp"

#define L_stat 100

using std::cerr;
using std::cout;
using std::endl;
using torch::Tensor;
using torch::indexing::Ellipsis;
using torch::indexing::Slice;

const utils::indices top{0,Ellipsis};
const utils::indices bottom{-1,Ellipsis};
const utils::indices left{Slice(1, -1), 0, Ellipsis};
const utils::indices right{Slice(1, -1), -1, Ellipsis};

const Tensor W = torch::tensor(
  {4.0/ 9.0,
    1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

const Tensor E = torch::tensor(
  {{0.0, 1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
   {0.0, 0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0}},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));
const torch::Tensor E_E = (E*E).sum_to_size({1,9}).clone().detach();
const Tensor E_rep = E.unsqueeze(0).unsqueeze(0).repeat({L_stat,L_stat,1,1});

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
const Tensor M = M_original.clone().detach();
//const Tensor Mi = M_original.inverse().clone().detach();
const Tensor M_rep = M.unsqueeze(0).unsqueeze(0).repeat({L_stat,L_stat,1,1});

const Tensor Mi = (1.0/36.0)*torch::tensor(
  {{  4.0, -4.0,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0},
   {  4.0, -1.0, -2.0,  6.0, -6.0,  0.0,  0.0,  9.0,  0.0},
   {  4.0, -1.0, -2.0,  0.0,  0.0,  6.0, -6.0, -9.0,  0.0},
   {  4.0, -1.0, -2.0, -6.0,  6.0,  0.0,  0.0,  9.0,  0.0},
   {  4.0, -1.0, -2.0,  0.0,  0.0, -6.0,  6.0, -9.0,  0.0},
   {  4.0,  2.0,  1.0,  6.0,  3.0,  6.0,  3.0,  0.0,  9.0},
   {  4.0,  2.0,  1.0, -6.0, -3.0,  6.0,  3.0,  0.0, -9.0},
   {  4.0,  2.0,  1.0, -6.0, -3.0, -6.0, -3.0,  0.0,  9.0},
   {  4.0,  2.0,  1.0,  6.0,  3.0, -6.0, -3.0,  0.0, -9.0}},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

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

const Tensor kernel_partial_y = -3.0*torch::tensor(
  {{ 1.0/36.0,  1.0/9.0,  1.0/36.0},
   {      0.0,      0.0,       0.0},
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
  const double r_tau;
  const double b_tau;
  const double s1, s2, s3, t2, t3;

  double init_s1(double r_tau, double b_tau)
  { return 2.0*r_tau*b_tau/(r_tau+b_tau); }

  double init_s2(double r_tau, double s1, double delta)
  { return 2.0*(r_tau-s1)/delta; }

  double init_s3(double s2, double delta)
  { return -s2/(2.0*delta); }

  double init_t2(double b_tau, double s1, double delta)
  { return 2.0*(s1-b_tau)/delta; }

  double init_t3(double t2, double delta)
  { return t2/(2.0*delta); }

  double eval(double psi) // Scalar function to evaluate the relaxation parameter
  {
    if (psi > delta) return r_tau;
    else if (delta >= psi && psi > 0) return s1 + s2*psi + s3*psi*psi;
    else if (0 >= psi && psi >= -delta) return s1 + t2*psi + t3*psi*psi;
    return b_tau;
  }

public:
  relaxation_function(double r_omega, double b_omega,  double delta):
  delta{delta},
  r_tau{1.0/r_omega},
  b_tau{1.0/b_omega},
  s1{init_s1(r_tau, b_tau)},
  s2{init_s2(/*r_tau=*/r_tau, /*s1=*/s1, delta)},
  s3{init_s3(s2, delta)},
  t2{init_t2(/*b_tau=*/b_tau, s1, delta)},
  t3{init_t3(t2, delta)}
  {
    // k_nu: kinematic viscosity
    // k_cs2: squared numeric sound speed
    std::cout << "\ns_nu parameters" << "\n"
      << "delta=" << delta << "\n"
      << "r_tau=" << r_tau << "\n"
      << "b_tau=" << b_tau << "\n"
      << "s1=" << s1 << "\n"
      << "s2=" << s2 << "\n"
      << "s3=" << s3 << "\n"
      << "t2=" << t2 << "\n"
      << "t3=" << t3 << std::endl;
  }

  void eval(torch::Tensor &s_nu, const torch::Tensor &psi_)
  {
    auto psi = psi_.squeeze(-1).clone().detach();
    auto bmask = (psi > delta);
    s_nu.masked_fill_(bmask, r_tau);

    bmask.copy_( (delta >= psi) * (psi > 0.0) );
    auto elements = s1 + s2*psi + s3*psi*psi;
    s_nu = torch::where(bmask, elements, s_nu);

    bmask.copy_( (0.0 >= psi) * (psi >= -delta) );
    elements.copy_(s1 + t2*psi + t3*psi*psi);
    s_nu = torch::where(bmask, elements, s_nu);

    bmask.copy_( psi < -delta );
    s_nu.masked_fill_(bmask, b_tau);
  }
};

class color
{
public:
  Tensor rho;
  Tensor f;
  Tensor coll_f;
  Tensor reco_f;
  Tensor mrtp_f;
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
  const double beta;

  color(int L, double R, double rho_0, double alpha, double nu, double A, bool invert_sigmoid=false):
  alpha{alpha},
  cs2{init_cs2(alpha)},
  nu{nu},
  rho_0{rho_0},
  omega{init_omega(nu, cs2)},
  A{A},
  beta{init_beta(invert_sigmoid)},
  phi{init_phi(alpha)},
  equ_factor{init_equ_factor(alpha)}
  {
    rho = torch::zeros({L, L, 1}, torch::kCUDA);
    init_rho(rho, rho_0, L, R, invert_sigmoid);
    f = torch::zeros({L, L, 9}, torch::kCUDA);
    f.copy_(1.0*equilibrium(torch::zeros({L,L,2}, torch::kCUDA)));
    mrtp_f = torch::zeros_like(f);
    reco_f = torch::zeros_like(f);
    coll_f = torch::zeros_like(f);
    m_eq = torch::zeros_like(f);

    C = torch::zeros_like(f);

    Tensor temp_s = torch::diagflat(torch::tensor({{0.0, s_e, s_zeta, 0.0, s_q, 0.0, s_q, 0.0, 0.0}}, torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA)));
    S = temp_s.unsqueeze(0).unsqueeze(0).repeat({L, L, 1, 1});
  }

  void step
  (
    const Tensor &rho_mix,
    const Tensor &rho_other,
    const Tensor &u,
    const Tensor &F,
    const Tensor &s_nu,
    const Tensor &n
  )
  {
    collision(rho_mix, rho_other, u, F, s_nu, n);
    solver::advect(f, coll_f);
  }



private:

  void collision
  (
    const Tensor &rho_mix,
    const Tensor &rho_other,
    const Tensor &u,
    const Tensor &F,
    const Tensor &s_nu,
    const Tensor &n
  )
  {
    reco_operator(reco_f, rho_mix, rho_other, u, F, s_nu, n);
    coll_f.copy_(f + reco_f);
  }

  void reco_operator
  (
    Tensor &Omega,
    const Tensor &rho_mix,
    const Tensor &rho_other,
    const Tensor &u,
    const Tensor &F,
    const Tensor &s_nu,
    const Tensor &n
  )
  {
    Tensor rho_ratio = rho/rho_mix;
    mrtp_operator(mrtp_f, u, F, s_nu);
    Tensor A = torch::mul(beta*rho_ratio*rho_other*torch::matmul(n,E),W);
    Omega.copy_(rho_ratio*mrtp_f + A);
  }

  void mrtp_operator(Tensor &Omega, const Tensor &u, const Tensor &F, const Tensor &s_nu)
  {
    // [(fM - m^eq + F)S + C]M^-1   (all matrices are tranposed)
    update_m_eq(u);
    update_C(u, s_nu);
    update_S(s_nu);

    Omega.copy_(
      (S.transpose(2,3).matmul(( f.matmul(M) - m_eq + A*(1.0 - 0.5*omega)*F ).unsqueeze(-1)).squeeze(-1) + C)
      .matmul(Mi)
    );

    /*
    Tensor EQU = equilibrium(u);
    const int Rows = f.size(0);
    const int Cols = f.size(1);
    for (int r=0; r<Rows; r++)
    {
      for (int c=0; c<Cols; c++)
      {
        Tensor SUM = M.matmul(f.index({r,c,Ellipsis}) - EQU.index({r,c,Ellipsis}));

        Omega.index_put_({r,c,Ellipsis},
                         Mi.matmul(
                            S.index({r,c,Ellipsis}).matmul(SUM)
                            + C.index({r,c,Ellipsis})
                            + A*(1.0 - 0.5*omega)*F.index({r,c,Ellipsis})
                          )
                        );
      }
    }*/

  }

  void update_S(const Tensor &s_nu)
  {
    S.index({Ellipsis, 7, 7}) = s_nu.squeeze(-1).clone().detach();
    S.index({Ellipsis, 8, 8}) = s_nu.squeeze(-1).clone().detach();
  }

  void update_C(const Tensor &u, const Tensor &s_nu)
  {
    const double a = 1.8*alpha - 0.8;
    Tensor Qx = a*rho.squeeze(-1)*u.index({Ellipsis,0});
    Tensor Qy = a*rho.squeeze(-1)*u.index({Ellipsis,1});

    C.index({Ellipsis,1}) = 3.0*(1.0 - 0.5*s_e)*(partial.x(Qx) + partial.y(Qy)).clone().detach();
    C.index({Ellipsis,7}) = (1.0 - 0.5*s_nu.squeeze(-1))*(partial.x(Qx) - partial.y(Qy)).clone().detach();

  }

  void update_m_eq(const Tensor &u)
  {
    #define ux u.index({Ellipsis,0})
    #define uy u.index({Ellipsis,1})
    Tensor u_u = (u*u).sum_to_size(rho.sizes()).squeeze(-1);
    m_eq.index({Ellipsis,0}) = 1.0;
    m_eq.index({Ellipsis,1}) = (-3.6*alpha - 0.4 + 3.0*u_u).clone().detach();
    m_eq.index({Ellipsis,2}) = (5.4*alpha - 1.4 - 3.0*u_u).clone().detach();
    m_eq.index({Ellipsis,3}) = ux.clone().detach();
    m_eq.index({Ellipsis,4}) = (-1.8*alpha - 0.2)*ux.clone().detach();
    m_eq.index({Ellipsis,5}) = uy.clone().detach();
    m_eq.index({Ellipsis,6}) = (-1.8*alpha - 0.2)*uy.clone().detach();
    m_eq.index({Ellipsis,7}) = (ux*ux - uy*uy).clone().detach();
    m_eq.index({Ellipsis,8}) = (ux*uy).clone().detach();
    m_eq.copy_(rho*m_eq);
  }

  Tensor equilibrium
  (
    const torch::Tensor &u
  )
  {
    torch::Tensor u_u = (u*u).sum_to_size(rho.sizes());
    torch::Tensor E_u = torch::matmul(u, E);
    return (rho*(phi + torch::mul(3.0*E_u*equ_factor + 4.5*E_u.pow(2) - 1.5*u_u, W))).clone().detach();
  }

  const Tensor phi;
  const Tensor equ_factor;

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
  double init_beta(bool invert)
  {
    if (invert) return 0.7;
    return -0.7;
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

std::ostream& operator<<(std::ostream& os, const color& c)
{
    return os << " color parameters" << "\n"
    << "alpha=" << c.alpha << "\n"
    << "cs2=" << c.cs2 << "\n"
    << "nu=" << c.nu << "\n"
    << "rho_0=" << c.rho_0 << "\n"
    << "omega=" << c.omega << "\n"
    << "A=" << c.A << "\n"
    << "beta=" << c.beta << endl;
}

void periodic_boundary_condition(Tensor &adv_f, const Tensor &col_f)
{
  // Complete the post-advection population using the post-collision populations
  adv_f.index(left) = col_f.index(right);
  adv_f.index(right) = col_f.index(left);
  adv_f.index(top) = col_f.index(bottom);
  adv_f.index(bottom) = col_f.index(top);
}

void build_F(Tensor &F, const Tensor &u, const Tensor &Fs);

void normalize_grad(Tensor &n, const Tensor &grad, Tensor &norm, double eps=1e-12)
{
  norm = torch::where(norm == 0, torch::full_like(norm, eps), norm);
  n.copy_(grad.div(norm));
}

int main(int argc, char *argv[])
{
  torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(torch::kDouble));
  if (!torch::cuda::is_available())
  {
    cerr << "CUDA is NOT available\n";
  }
  const torch::Device dev = torch::kCUDA;

  // Parameters
  const int L = 100; // Domain size
  const double R = 25.0; // Droplet radius

  const double sigma = 0.1; // Interfacial tension coefficient

  // Initialization
  color r{L, R, /*rho_0=*/1.0, /*alpha=*/0.2, /*nu=*/0.1667, /*A=*/0.5, true};
  cout << "RED" << r << endl;
  color b{L, R, /*rho_0=*/1.0, /*alpha=*/0.2, /*nu=*/0.1667, /*A=*/0.5};
  cout << "BLUE" << b << endl;
  relaxation_function s_nu_function{r.omega, b.omega, 0.1};
  Tensor u = torch::zeros({L, L, 2}, dev);
  Tensor rho_mix = torch::zeros({L, L, 1}, dev);
  Tensor s_nu = torch::zeros({L, L}, dev);
  Tensor rho_n = torch::zeros({L, L, 1}, dev);
  Tensor grad_rho_n = torch::zeros({L, L, 2}, dev);
  Tensor grad_norm = torch::zeros({L,L,1},dev);
  Tensor n = torch::zeros({L,L,2}, dev);
  Tensor K = torch::zeros({L,L,1}, dev);
  Tensor F_s = torch::zeros({L,L,2}, dev);
  Tensor F = torch::zeros({L,L,9}, dev);

  rho_mix.copy_(r.rho + b.rho);

  const int T = 100;
  // Results
  Tensor r_fs = torch::zeros({L,L,9,T});
  Tensor b_fs = torch::zeros({L,L,9,T});
  Tensor uxs = torch::zeros({L,L,T});
  Tensor uys = torch::zeros_like(uxs);
  Tensor rhos = torch::zeros_like(uxs);
  Tensor rhons = torch::zeros_like(uxs);
  Tensor nxs = torch::zeros_like(uxs);
  Tensor nys = torch::zeros_like(uxs);
  Tensor Ks = torch::zeros_like(uxs);
  Tensor Fsxs = torch::zeros_like(uxs);
  Tensor Fsys = torch::zeros_like(uxs);
  Tensor norms = torch::zeros_like(uxs);
  Tensor gradxs = torch::zeros_like(uxs);
  Tensor gradys = torch::zeros_like(uxs);
  Tensor s_nus = torch::zeros_like(uxs);
  // Main loop
  cout << "main loop start" << endl;
  cout << torch::tensor({0.0}) << endl;
  for (int t=0; t < T; t++)
  {
    // Compute equilibrium distributions
    eval_phase_field(rho_n, r.rho, r.rho_0, b.rho, b.rho_0);
    rhons.index({Ellipsis,t}) = rho_n.squeeze(2).clone().detach();
    partial.grad(grad_rho_n, rho_n);
    //grad_rho_n = torch::where(torch::abs(grad_rho_n) <= 1e-6,
    //                         torch::full_like(grad_rho_n, 0.0), grad_rho_n);
    gradxs.index({Ellipsis,t}) = grad_rho_n.index({Ellipsis,0}).clone().detach();
    gradys.index({Ellipsis,t}) = grad_rho_n.index({Ellipsis,1}).clone().detach();
    grad_norm.copy_(
      torch::sqrt(grad_rho_n.index({Ellipsis, 0}).pow(2)
                  + grad_rho_n.index({Ellipsis, 1}).pow(2)).unsqueeze(-1)
       );
    //grad_norm = torch::where(torch::abs(grad_norm) <= 1e-3,
    //                         torch::full_like(grad_norm, 0.0), grad_norm);
    norms.index({Ellipsis,t}) = grad_norm.index({Ellipsis,0}).clone().detach();

    //n = grad_rho_n/torch::norm(rho_n, 2, -1).unsqueeze(-1);
    //normalize_grad(n, grad_rho_n, grad_norm);
    n = -torch::where(grad_norm >= 1e-1,
                     torch::nn::functional::normalize(grad_rho_n,
                                         torch::nn::functional::NormalizeFuncOptions()
                                         .p(2).dim(-1)),
                     torch::full_like(grad_rho_n, 0.0));
    //n = torch::where(torch::abs(n) <= 1e-1, torch::full_like(n, 0.0), n);
    nxs.index({Ellipsis,t}) = n.index({Ellipsis, 0}).clone().detach();
    nys.index({Ellipsis,t}) = n.index({Ellipsis, 1}).clone().detach();
    eval_local_curvature(K,-n);
    Ks.index({Ellipsis,t}) = K.squeeze(-1);
    F_s.copy_( 0.5*sigma*K*grad_rho_n );
    Fsxs.index({Ellipsis,t}) = F_s.index({Ellipsis,0});
    Fsys.index({Ellipsis,t}) = F_s.index({Ellipsis,1});
    build_F(F, u, F_s);
    s_nu_function.eval(s_nu, rho_n);
    s_nu.pow_(-1.0);
    s_nus.index({Ellipsis,t}) = s_nu.detach().clone();
    // Step: collision and advection
    r.step(/*rho_mix=*/rho_mix, /*rho_other=*/b.rho, /*u=*/u, /*F=*/F, /*s_nu=*/s_nu, /*n=*/n);
    b.step(rho_mix, r.rho, u, F, s_nu, n);
    // Boundary conditions
    periodic_boundary_condition(r.f, r.coll_f);
    r_fs.index_put_({Ellipsis,t}, r.f);
    periodic_boundary_condition(b.f, b.coll_f);
    b_fs.index_put_({Ellipsis,t}, b.f);
    // Compute macroscopics quantities
    solver::calc_rho(r.rho, r.f);
    solver::calc_rho(b.rho, b.f);
    rho_mix.copy_(r.rho + b.rho);

    solver::calc_u(u, r.f + b.f, rho_mix);
    u.copy_(u + 0.5*F_s/rho_mix);
    // Save snapshots
    uxs.index({Ellipsis,t}) = u.index({Ellipsis, 0}).clone().detach();
    uys.index({Ellipsis,t}) = u.index({Ellipsis, 1}).clone().detach();
    rhos.index({Ellipsis,t}) = rho_mix.squeeze(2).clone().detach();
  }

  cout << "Saving results" << endl;
  torch::save(r_fs, "static-droplet-r-fs.pt");
  torch::save(b_fs, "static-droplet-b-fs.pt");
  torch::save(uxs, "static-droplet-ux.pt");
  torch::save(uys, "static-droplet-uy.pt");
  torch::save(nxs, "static-droplet-nx.pt");
  torch::save(nys, "static-droplet-ny.pt");
  torch::save(rhos, "static-droplet-rho.pt");
  torch::save(rhons, "static-droplet-rhon.pt");
  torch::save(Ks, "static-droplet-ks.pt");
  torch::save(norms, "static-droplet-norms.pt");
  torch::save(Fsxs, "static-droplet-fx.pt");
  torch::save(Fsys, "static-droplet-fy.pt");
  torch::save(gradxs, "static-droplet-gradx.pt");
  torch::save(gradys, "static-droplet-grady.pt");
  torch::save(s_nus, "static-droplet-s_nus.pt");

  return 0;
}

void build_F(Tensor &F, const Tensor &u, const Tensor &Fs)
{
  //#define ux u.index({Ellipsis,0})
  //#define uy u.index({Ellipsis,1})
  #define Fsx Fs.index({Ellipsis,0})
  #define Fsy Fs.index({Ellipsis,1})
/*
  F.index({Ellipsis,0}) = 0.0;
  F.index({Ellipsis,1}) = 6.0*(ux*Fsx + uy*Fsy).clone().detach();
  F.index({Ellipsis,2}) = -6.0*(ux*Fsx + uy*Fsy).clone().detach();
  F.index({Ellipsis,3}) = Fsx.clone().detach();
  F.index({Ellipsis,4}) = -1.0*Fsx.clone().detach();
  F.index({Ellipsis,5}) = Fsy.clone().detach();
  F.index({Ellipsis,6}) = -1.0*Fsy.clone().detach();
  F.index({Ellipsis,7}) = 2.0*(ux*Fsx - uy*Fsy).clone().detach();
  F.index({Ellipsis,8}) = (ux*Fsy + uy*Fsx).clone().detach();
  */
  Tensor E_u = torch::matmul(u,E);

  Tensor A = 3.0*(E_rep - u.unsqueeze(-1)) + 9.0*(E_u.unsqueeze(-2)*E);
  Tensor B = A*Fs.unsqueeze(-1);
  Tensor C = B.sum(2);
  Tensor D = torch::mul(C,W);

  F = M_rep.matmul(D.unsqueeze(-1)).squeeze(-1).clone().detach();

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
