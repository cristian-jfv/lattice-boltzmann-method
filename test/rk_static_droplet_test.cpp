#include <iostream>
#include <limits>
#include <torch/torch.h>

#include "../src/solver.hpp"
#include "../src/utils.hpp"

#define L 101
#define Radius 25.0

using torch::Tensor;
using torch::indexing::Ellipsis;
using torch::indexing::Slice;
namespace tnnf = torch::nn::functional;

using std::cout;
using std::endl;

const auto dev = torch::kCUDA;
const double cs2 = 1.0/3.0; // numerical speed of sound
const double ics2 = 3.0;

const utils::indices top{0,Ellipsis};
const utils::indices v_top{1,Ellipsis};

const utils::indices bottom{-1,Ellipsis};
const utils::indices v_bottom{-2,Ellipsis};

const utils::indices left{Slice(1, -1), 0, Ellipsis};
const utils::indices v_left{Slice(1, -1), 1, Ellipsis};

const utils::indices right{Slice(1, -1), -1, Ellipsis};
const utils::indices v_right{Slice(1, -1), -2, Ellipsis};

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
    ans.index({Ellipsis,0}) = x(psi).clone().detach();
    ans.index({Ellipsis,1}) = y(psi).clone().detach();
  }

};
partial_derivatives partial{};

class colour
{
public:
  const double rho_0;
  const double alpha;
  const double A;
  const double omega_rp;
  const double cks2;
  const double beta;

  const Tensor phi;
  const Tensor xi;
  const Tensor zero_u;
  const Tensor B = torch::tensor(
  {-4.0/27.0,
    2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0,
    5.0/108.0, 5.0/108.0, 5.0/108.0, 5.0/108.0},
  torch::TensorOptions().dtype(torch::kDouble).device(dev));

  Tensor omega1; // Result of the standard single phase operator
  Tensor omega2; // Result of the perturbation operator
  Tensor omega3; // Result of the redistribution operator
  Tensor equ_f; // Stores the equilibrium distribution function
  Tensor col_f; // Stores the dist. function after the collision step
  Tensor adv_f; // Stores the dist. funciton after the advection step
  Tensor rho;

  colour(int R, int C, double rho_0, double alpha, double A, double nu, double beta):
  rho_0{rho_0},
  alpha{alpha},
  A{A},
  omega_rp{init_omega(nu)},
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
    rho = torch::zeros({R,C}, dev);
  }

  void step
  (
    const Tensor &u,
    const Tensor &relax_params,
    const Tensor &eta,
    const Tensor &kappa,
    const Tensor &rho_mix,
    const Tensor &F,
    const Tensor &F_norm
  )
  {
    eval_omega3
      (
        /*omega_=*/omega3,
        /*u=*/u,
        /*relax_params=*/relax_params,
        /*eta=*/eta,
        /*kappa=*/kappa,
        /*rho_mix=*/rho_mix, F, F_norm);
    col_f.copy_(
      adv_f + omega3
    );
    solver::advect(adv_f, col_f);
    apply_boundary_conditions(adv_f, col_f, u);
  }

  void eval_equilibrium(Tensor &omega_, const Tensor &rho_, const Tensor &u)
  {
    Tensor E_u = torch::matmul(u,E);
    Tensor u_u = (u*u).sum(-1).unsqueeze(-1);

    // Tensor A = ics2*E_u*xi;
    // Tensor B = 0.5*ics2*ics2*E_u.pow(2);
    // Tensor C = 0.5*ics2*u_u;
    // Tensor D = torch::mul(A + B - C, W);

    omega_.copy_(
      rho_.unsqueeze(-1)*(
        phi
        + torch::mul(ics2*E_u + 0.5*ics2*ics2*E_u.pow(2) - 0.5*ics2*u_u, W)
      )
    );
  }


private:

  void apply_boundary_conditions(Tensor &adv_f, const Tensor &col_f, const Tensor &u)
  {
    // Complete the post-advection population using the post-collision populations
    adv_f.index(left) = col_f.index(right).clone().detach();
    adv_f.index(right) = col_f.index(left).clone().detach();
    adv_f.index(top) = col_f.index(bottom).clone().detach();
    adv_f.index(bottom) = col_f.index(top).clone().detach();
  }

  void eval_omega3
  (
    Tensor &omega_,
    const Tensor &u,
    const Tensor &relax_params,
    const Tensor &eta,
    const Tensor &kappa,
    const Tensor &rho_mix,
    const Tensor &F,
    const Tensor &F_norm
  )
  {
    eval_omega1(omega1, rho, u, relax_params);
    //eval_omega2(omega2, relax_params, eta);
    eval_reis_omega2(omega2, F, F_norm);
    Tensor rho_ratio = rho/rho_mix;

    // Tensor A = rho_ratio.unsqueeze(-1)*(omega1 + omega2);

    omega_.copy_(
      // beta must be initialised with the appropiate sign
      // rho_ratio.unsqueeze(-1)*(omega1 + omega2) + beta*kappa
      omega1 + omega2
    );
  }

  void eval_reis_omega2(Tensor &omega_, const Tensor &F, const Tensor &F_norm)
  {
    omega_.copy_(
        0.5*A*F_norm.unsqueeze(-1)*(
      torch::mul( torch::matmul(F,E).pow(2.0)/(1e-20 + F_norm.pow(2.0).unsqueeze(-1)) ,W) - B )
    );
  }

  void eval_omega2(Tensor &omega_, const Tensor &relax_params, const Tensor &eta)
  {
    // Tensor B = 1.0 - 0.5*relax_params.unsqueeze(-1);
    omega_.copy_(
      A*(1.0 - 0.5*relax_params.unsqueeze(-1))*eta
    );
  }

  void eval_omega1(Tensor &omega_, const Tensor &rho_, const Tensor &u, const Tensor &relax_params)
  {
    // TODO: Single or position wise relaxation parameter?
    eval_equilibrium(equ_f, rho_, u);
    omega_.copy_(
      relax_params.unsqueeze(-1)*(equ_f - adv_f)
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

double sigmoid(double x) { return 1.0/(1.0 + std::exp(-x)); }

void init_rho(torch::Tensor &rho, double rho_0, double L_, double R, bool invert)
{
  const int rows = rho.size(0);
  const int cols = rho.size(1);
  const double C = L_/2.0;
  //const double h = 9.0;
  const double factor = 2.0;

  for(int r=0; r<rows; r++)
  {
    for(int c=0; c<cols; c++)
    {
      double s = std::sqrt((r-C)*(r-C) + (c-C)*(c-C));
      double ans = 0.0;
      if(invert)
      {
        // Fill the droplet
        //if ( s < R) ans = 1.0;
        //else if ( R <= s && s <= (R+h) )
        ans = 1.0 - sigmoid(factor*(s-R));
        //else ans = 0;
      }
      else
      {
        // Fill the sourrounding fluid
        //if ( s < R ) ans = 0.0;
        //else if ( R <= s && s <= (R+h) )
        ans = sigmoid(factor*(s-R));
        //else ans = 1.0;
      }
      rho[r][c] = rho_0*ans;
    }
  }
}

void eval_eta(Tensor &eta, const Tensor &u, const Tensor &Fs)
{
  // eta is defined as the color independent part of the perturbation operator
  Tensor E_u = torch::matmul(u,E);
  // TODO: define factor for the second term in the expression
  // Tensor A = ics2*(E_rep - u.unsqueeze(-1));
  // Tensor B = ics2*(E_u.unsqueeze(-2)*E);
  // Tensor C = A+B;
  // Tensor D = ((A+B)*Fs.unsqueeze(-1)).sum(2);

  eta.copy_(
    torch::mul(
    ((ics2*(E_rep - u.unsqueeze(-1)) + ics2*(E_u.unsqueeze(-2)*E))*Fs.unsqueeze(-1)).sum(2)
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
  //Tensor A = (r_rho*b_rho/rho.pow(2)).unsqueeze(-1);
  //Tensor B = (r_rho.unsqueeze(-1)*r_phi + b_rho.unsqueeze(-1)*b_phi).squeeze(-1);
  //Tensor C = torch::matmul(n,E);
  //Tensor D = A*C;

  kappa.copy_(
    (r_rho*b_rho/rho/*.pow(2)*/).unsqueeze(-1)
    *torch::mul(torch::matmul(-n,E),W)
    //*(r_rho.unsqueeze(-1)*r_phi + b_rho.unsqueeze(-1)*b_phi).squeeze(-1)
  );
  //cout << kappa.max() << endl;
  //cout << kappa.min() << endl;
}

void eval_local_curvature(Tensor &K, const Tensor &n)
{
  K = (
    n.index({Ellipsis,0})*n.index({Ellipsis,1})*( partial.y(n.index({Ellipsis,0})) + partial.x(n.index({Ellipsis,1})) )
      - n.index({Ellipsis,0}).pow(2.0)*partial.y(n.index({Ellipsis,1})) - n.index({Ellipsis,1}).pow(2.0)*partial.x(n.index({Ellipsis,0}))
  ).clone().detach();
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
  // rho_n = ( (r_rho - b_rho)/(r_rho + b_rho) ).clone().detach();
}

std::ostream& operator<<(std::ostream& os, const colour& c)
{
    return os << " color parameters" << "\n"
    << "alpha=" << c.alpha << "\n"
    << "cks2=" << c.cks2 << "\n"
    //<< "nu=" << c.nu << "\n"
    << "rho_0=" << c.rho_0 << "\n"
    << "omega=" << c.omega_rp << "\n"
    << "A=" << c.A << "\n"
    << "beta=" << c.beta << endl;
}

int main()
{
  torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(torch::kDouble));
  if (!torch::cuda::is_available())
  {
    std::cerr << "CUDA is NOT available\n";
  }
  double minDouble = std::numeric_limits<double>::min();
  std::cout << "Smallest positive double: " << minDouble << std::endl;

  // Macroscopic parameters
  Tensor u = torch::zeros({L,L,2}, dev);
  u = 1e-15*u.normal_().clone().detach();
  Tensor rho_mix = torch::ones({L,L}, dev);

  // Colour-independet quantities
  Tensor eta = torch::zeros({L,L,9}, dev);
  Tensor kappa = torch::zeros({L,L,9}, dev);
  Tensor relax_params = torch::zeros({L,L},dev);

  // Interfacial tension
  const double sigma = 5e-3;
  Tensor phase_field = torch::zeros({L,L}, dev);
  Tensor grad_pf = torch::zeros({L,L,2}, dev);
  Tensor grad_norm = torch::zeros({L,L}, dev);
  Tensor K = torch::zeros({L,L}, dev);
  Tensor n = torch::zeros({L,L,2}, dev);
  Tensor Fs = torch::zeros({L,L,2}, dev);

  colour r{/*R=*/L, /*C=*/L, /*rho_0=*/1.2, /*alpha=*/1.0/3.0, /*A=*/1e-4, /*nu=*/0.16, /*beta=*/+0.7};
  cout << "RED" << r << endl;
  colour b{/*R=*/L, /*C=*/L, /*rho_0=*/1.0, /*alpha=*/0.2, /*A=*/1e-4, /*nu=*/0.14, /*beta=*/-0.7};
  cout << "BLUE" << b << endl;
  // Initialise blue and red densities
  init_rho(r.rho, r.rho_0, L, Radius, true);
  r.eval_equilibrium(r.adv_f, r.rho, u);
  init_rho(b.rho, b.rho_0, L, Radius, false);
  b.eval_equilibrium(b.adv_f, b.rho, u);
  r.rho.copy_( r.adv_f.sum(2) );
  b.rho.copy_( b.adv_f.sum(2) );
  rho_mix.copy_(r.rho + b.rho);

  relaxation_function relax_func{r.omega_rp, b.omega_rp, /*delta=*/0.98};

  const int T = 2000;

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
  Tensor rparams = torch::zeros_like(uxs);
  Tensor kappas = torch::zeros_like(r_fs);
  Tensor omega1s = torch::zeros_like(r_fs);
  Tensor omega2s = torch::zeros_like(r_fs);
  Tensor omega3s = torch::zeros_like(r_fs);

  cout << "main loop" << endl;
  cout << torch::tensor({0}) << endl;
  for (int t=0; t < T; t++)
  {
    // Calculate interfacial tension
    eval_phase_field(phase_field, r.rho, r.rho_0, b.rho, b.rho_0);
    rhons.index({Ellipsis,t}) = phase_field.clone().detach();

    partial.grad(grad_pf, phase_field);
    grad_norm.copy_(
      torch::sqrt(grad_pf.index({Ellipsis, 0}).pow(2)
                  + grad_pf.index({Ellipsis, 1}).pow(2))
    );
    norms.index({Ellipsis,t}) = grad_norm.clone().detach();

    gradxs.index({Ellipsis,t}) = grad_pf.index({Ellipsis,0}).clone().detach();
    gradys.index({Ellipsis,t}) = grad_pf.index({Ellipsis,1}).clone().detach();
    Tensor copy_grad_pf = torch::where
    (
      grad_norm.unsqueeze(-1) <= 0.1*grad_norm.max(),
      torch::full_like(grad_pf, 0.0), grad_pf
    );
    n = -tnnf::normalize(copy_grad_pf, tnnf::NormalizeFuncOptions().p(2).dim(-1));
    //n.copy_(
    //  grad_pf/(1e5*grad_norm.min() + grad_norm.unsqueeze(-1))
    //);
    nxs.index({Ellipsis,t}) = n.index({Ellipsis, 0}).clone().detach();
    nys.index({Ellipsis,t}) = n.index({Ellipsis, 1}).clone().detach();

    eval_local_curvature(K, n);
    Ks.index({Ellipsis,t}) = K.clone().detach();

    Fs.copy_( 0.5*sigma*K.unsqueeze(-1)*grad_pf ); // TODO: plus or minus sign?
    Fsxs.index({Ellipsis,t}) = Fs.index({Ellipsis,0});
    Fsys.index({Ellipsis,t}) = Fs.index({Ellipsis,1});

    eval_eta(eta, u, Fs);

    eval_kappa
      (/*kappa=*/kappa,
       /*n=*/n,
       /*rho=*/rho_mix,
       /*r_rho=*/r.rho,
       /*r_phi=*/r.phi, b.rho, b.phi);
    kappas.index({Ellipsis,t}) = kappa.clone().detach();
    relax_func.eval(relax_params, phase_field);
    relax_params.pow_(-1);
    rparams.index({Ellipsis,t}) = relax_params.clone().detach();

    // Time step
    r.step(u, relax_params, eta, kappa, rho_mix, grad_pf, grad_norm);
    r_fs.index_put_({Ellipsis,t}, r.adv_f);
    omega1s.index_put_({Ellipsis,t}, r.omega1);
    omega2s.index_put_({Ellipsis,t}, r.omega2);
    omega3s.index_put_({Ellipsis,t}, r.omega3);
    b.step(u, relax_params, eta, kappa, rho_mix, grad_pf, grad_norm);
    b_fs.index_put_({Ellipsis,t}, b.adv_f);

    // solver::calc_rho(r.rho, r.adv_f);
    // solver::calc_rho(b.rho, b.adv_f);
    r.rho.copy_( r.adv_f.sum(2) );
    b.rho.copy_( b.adv_f.sum(2) );
    rho_mix.copy_(r.rho + b.rho);
    //r.eval_equilibrium(r.equ_f, r.rho, u);
    //r.eval_equilibrium(b.equ_f, b.rho, u);


    solver::calc_u(u, r.adv_f + b.adv_f, rho_mix.unsqueeze(-1));
    //u = (u + 0.5*Fs).clone().detach();

    uxs.index({Ellipsis,t}) = u.index({Ellipsis, 0}).clone().detach();
    uys.index({Ellipsis,t}) = u.index({Ellipsis, 1}).clone().detach();
    rhos.index({Ellipsis,t}) = rho_mix.clone().detach();
  }
  cout << "saving results" << endl;
  torch::save(r_fs,   "rk-static-droplet-r-fs.pt");
  torch::save(b_fs,   "rk-static-droplet-b-fs.pt");
  torch::save(uxs,    "rk-static-droplet-ux.pt");
  torch::save(uys,    "rk-static-droplet-uy.pt");
  torch::save(nxs,    "rk-static-droplet-nx.pt");
  torch::save(nys,    "rk-static-droplet-ny.pt");
  torch::save(rhos,   "rk-static-droplet-rho.pt");
  torch::save(rhons,  "rk-static-droplet-rhon.pt");
  torch::save(Ks,     "rk-static-droplet-ks.pt");
  torch::save(norms,  "rk-static-droplet-norms.pt");
  torch::save(Fsxs,   "rk-static-droplet-fx.pt");
  torch::save(Fsys,   "rk-static-droplet-fy.pt");
  torch::save(gradxs, "rk-static-droplet-gradx.pt");
  torch::save(gradys, "rk-static-droplet-grady.pt");
  torch::save(rparams, "rk-static-droplet-rparams.pt");
  torch::save(kappas, "rk-static-droplet-kappas.pt");
  torch::save(omega1s, "rk-static-droplet-omegas1.pt");
  torch::save(omega2s, "rk-static-droplet-omegas2.pt");
  torch::save(omega3s, "rk-static-droplet-omegas3.pt");
  return 0;
}
