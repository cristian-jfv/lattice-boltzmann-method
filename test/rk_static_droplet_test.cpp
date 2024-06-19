#include <iostream>
#include <torch/torch.h>

#include "../src/solver.hpp"
#include "../src/utils.hpp"

#define L 100

using torch::Tensor;
using torch::indexing::Ellipsis;
using torch::indexing::Slice;

using std::cout;
using std::endl;

const auto dev = torch::kCUDA;
const double cs2 = 1.0/3.0; // numerical speed of sound
const double ics2 = 3.0;

const utils::indices top{0,Ellipsis};
const utils::indices bottom{-1,Ellipsis};
const utils::indices left{Slice(1, -1), 0, Ellipsis};
const utils::indices right{Slice(1, -1), -1, Ellipsis};

const Tensor W = torch::tensor(
  {4.0/ 9.0,
    1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0},
  torch::TensorOptions().dtype(torch::kDouble).device(dev));

const Tensor E = torch::tensor(
  {{0.0, 1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
   {0.0, 0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0}},
  torch::TensorOptions().dtype(torch::kDouble).device(dev));

const Tensor E_rep = E.unsqueeze(0).unsqueeze(0).repeat({L,L,1,1}).clone().detach();

class colour
{
public:
  const double alpha;
  const double A;
  const double omega;
  const double cks2;
  const double beta;

  const Tensor phi;
  const Tensor xi;
  const Tensor zero_u;

  Tensor omega1; // Result of the standard single phase operator
  Tensor omega2; // Result of the perturbation operator
  Tensor omega3; // Result of the redistribution operator
  Tensor equ_f; // Stores the equilibirum distribution function
  Tensor col_f; // Stores the dist. function after the collision step
  Tensor adv_f; // Stores the dist. funciton after the advection step
  Tensor rho;

  colour(int R, int C, double alpha, double A, double nu, double beta):
  alpha{alpha},
  A{A},
  omega{init_omega(nu)},
  cks2{init_cks2(alpha)},
  beta{beta},
  phi{init_phi(alpha)},
  xi{init_xi(alpha)},
  zero_u{torch::zeros({R,C,2})}
  {
    adv_f = torch::zeros({R,C,9}, dev);
    col_f = torch::zeros_like(adv_f);
    equ_f = torch::zeros_like(adv_f);
    omega1 = torch::zeros_like(adv_f);
    omega2 = torch::zeros_like(adv_f);
    omega3 = torch::zeros_like(adv_f);
    rho = torch::zeros({R,C,1}, dev);
  }

  void collision_step
  (
    const Tensor &u,
    const Tensor &relax_params,
    const Tensor &eta,
    const Tensor &kappa,
    const Tensor &rho_mix
  )
  {
    eval_omega3(omega3, u, relax_params, eta, kappa, rho_mix);
    col_f.copy_(
      adv_f + omega3
    );
    solver::advect(adv_f, col_f);
    periodic_boundary_condition(adv_f, col_f);
  }

private:

  void periodic_boundary_condition(Tensor &adv_f, const Tensor &col_f)
  {
    // Complete the post-advection population using the post-collision populations
    adv_f.index(left) = col_f.index(right).clone().detach();
    adv_f.index(right) = col_f.index(left).clone().detach();
    adv_f.index(top) = col_f.index(bottom).clone().detach();
    adv_f.index(bottom) = col_f.index(top).clone().detach();
  }

  void eval_omega3
  (
    Tensor &omega,
    const Tensor &u,
    const Tensor &relax_params,
    const Tensor &eta,
    const Tensor &kappa,
    const Tensor &rho_mix
  )
  {
    eval_omega1(omega1, rho, u);
    eval_omega2(omega2, relax_params, eta);
    Tensor rho_ratio = rho/rho_mix;
    omega.copy_(
      // beta must be initialised with the appropiate sign
      rho_ratio*(omega1 + omega2) + beta*kappa
    );
  }

  void eval_omega2(Tensor &omega, const Tensor &relax_params, const Tensor &eta)
  {
    omega.copy_(
      A*(1.0 - 0.5*relax_params)*eta
    );
  }

  void eval_omega1(Tensor &omega, const Tensor &rho_, const Tensor &u)
  {
    // TODO: Single or position wise relaxation parameter?
    eval_equilibrium(equ_f, rho_, u);
    omega.copy_(
      omega*(equ_f - adv_f)
    );
  }

  void eval_equilibrium(Tensor &omega, const Tensor &rho_, const Tensor &u)
  {
    Tensor E_u = torch::matmul(u,E);
    Tensor u_u = (u*u).sum(2);
    omega.copy_(
      rho_*(phi + torch::mul(ics2*E_u*xi + 0.5*ics2*ics2*E_u.pow(2) - 0.5*ics2*u_u, W))
    );
  }

  double init_omega(double nu)
  { return 1.0/(0.5 + nu/cs2); }

  double init_cks2(double alpha)
  { return 0.6*(1.0 - alpha); }

  Tensor init_phi(double alpha)
  {
    double a = 0.2*(1 - alpha);
    double b = 0.05*(1 - alpha);
    return torch::tensor({alpha, a, a, a, a, b, b, b, b}, dev);
  }

  Tensor init_xi(double alpha)
  {
    double cks2 = init_cks2(alpha);
    Tensor E_E = (E*E).sum(0);

    return (1.0 + 0.5*(ics2*cks2 - 1.0)*(ics2*E_E - 4.0)).clone().detach();
  }

};

struct relaxation_function
{
private:
  const double delta;
  const double r_tau;
  const double b_tau;
  const double s1, s2, s3, t2, t3;

  double init_s1(double r_tau, double b_tau)
  { return 2.0*r_tau*b_tau/(r_tau + b_tau); }

  double init_s2(double r_tau, double s1, double delta)
  { return 2.0*(r_tau - s1)/delta; }

  double init_s3(double s2, double delta)
  { return -s2/delta; }

  double init_t2(double b_tau, double s1, double delta)
  { return 2.0*(s1 + b_tau)/delta; }

  double init_t3(double t2, double delta)
  { return -t2/delta; }

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

void eval_eta(Tensor &eta, const Tensor &u, const Tensor &Fs)
{
  // eta is defined as the color independent part of the perturbation operator
  Tensor E_u = torch::matmul(u,E);
  // TODO: define factor for the second term in the expression
  eta.copy_(
    torch::mul(
    (ics2*(E_rep - u.unsqueeze(-1)) + ics2*(E_u.unsqueeze(-2)*E))*Fs.unsqueeze(-1).sum(2)
    , W)
  );
}

void eval_kappa
(
  Tensor &kappa,
  const Tensor &n,
  const Tensor &rho,
  const Tensor &r_rho,
  const Tensor &r_phi,
  const Tensor &b_rho,
  const Tensor &b_phi
)
{
  kappa.copy_(
    (r_rho*b_rho/rho.pow(2)) * torch::matmul(n,E)*(r_rho*r_phi + b_rho*b_phi)
  );
}

void eval_phase_field
(
  Tensor &psi,
  const Tensor &r_rho,
  const Tensor &b_rho
);

void eval_K
(
  Tensor &K,
  const Tensor &n
);

int main()
{
  // Macroscopic parameters
  Tensor u = torch::zeros({L,L,2}, dev);
  Tensor rho_mix = torch::zeros({L,L}, dev);

  Tensor eta = torch::zeros({L,L,9}, dev);
  Tensor kappa = torch::zeros({L,L,9}, dev);

  Tensor phase_field = torch::zeros({L,L}, dev);
  Tensor grad_pf = torch::zeros({L,L,2}, dev);
  Tensor K = torch::zeros({L,L}, dev);
  Tensor n = torch::zeros({L,L,2}, dev);
  Tensor Fs = torch::zeros({L,L,2}, dev);

  colour r{/*R=*/L, /*C=*/L, /*alpha=*/0.2, /*A=*/0.5, /*nu=*/0.1667, /*beta=*/0.7};
  colour b{/*R=*/L, /*C=*/L, /*alpha=*/0.2, /*A=*/0.5, /*nu=*/0.1667, /*beta=*/-0.7};
  // TODO: initialise red and blue densities

  cout << "main loop" << endl;
  const int T = 100;
  for (int t=0; t < T; t++)
  {
  }

  return 0;
}
