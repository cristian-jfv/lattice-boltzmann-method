#include <ATen/TensorIndexing.h>
#include <c10/core/DeviceType.h>
#include <cmath>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <torch/serialize.h>
#include <torch/torch.h>
#include <toml++/toml.hpp>

#include "../src/solver.hpp"
#include "../src/colour.hpp"
#include "../src/differential.hpp"

using std::cout;
using std::endl;
using std::cerr;

using torch::indexing::Slice;
using torch::indexing::Ellipsis;
using torch::Tensor;

differential D{};

template<typename T>
T try_value(const toml::node_view<toml::node>& tbl, const std::string name)
{
  std::optional<T> op = tbl[name].template value<T>();
  if (op.has_value()) return op.value();
  else throw std::runtime_error(name + "not defined in parameters file");
}

struct relaxation_function
{
private:
  const double delta;
  const double r_omega;
  const double b_omega;
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

  double init_omega(double nu, double cs2)
  { return 1.0/( 0.5 + nu/cs2 ); }

public:
  relaxation_function(const colour red, const colour blue,  double delta):
  delta{delta},
  r_omega{init_omega(red.nu, red.cs2)},
  b_omega{init_omega(blue.nu, blue.cs2)},
  s1{init_s1(r_omega, b_omega)},
  s2{init_s2(/*r_tau=*/r_omega, /*s1=*/s1, delta)},
  s3{init_s3(s2, delta)},
  t2{init_t2(/*b_tau=*/b_omega, s1, delta)},
  t3{init_t3(t2, delta)}
  {
    // k_nu: kinematic viscosity
    // k_cs2: squared numeric sound speed
    std::cout << "\ns_nu parameters" << "\n"
      << "delta=" << delta << "\n"
      << "r_tau=" << r_omega << "\n"
      << "b_tau=" << b_omega << "\n"
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
    s_nu.masked_fill_(bmask, r_omega);

    bmask.copy_( (delta >= psi) * (psi > 0.0) );
    auto elements = s1 + s2*psi + s3*psi*psi;
    s_nu = torch::where(bmask, elements, s_nu);

    bmask.copy_( (0.0 >= psi) * (psi >= -delta) );
    elements.copy_(s1 + t2*psi + t3*psi*psi);
    s_nu = torch::where(bmask, elements, s_nu);

    bmask.copy_( psi < -delta );
    s_nu.masked_fill_(bmask, b_omega);
  }
};

struct domain
{
  const int R;
  const int C;
  const int T;
  const int nr_snapshots;
  const int period_snapshots;
  domain(const toml::node_view<toml::node>& tbl):
  R{try_value<int>(tbl, "rows")},
  C{try_value<int>(tbl, "columns")},
  T{try_value<int>(tbl, "time_steps")},
  nr_snapshots{try_value<int>(tbl, "nr_snapshots")},
  period_snapshots{(int)(T/nr_snapshots)}
  {}
};

std::ostream& operator<<(std::ostream& os, const domain& d)
{
  return os << "DOMAIN parameters:\n"
  << "R=" << d.R
  << "\nC=" << d.C
  << "\nT=" << d.T
  << "\nnr_snapshots=" << d.nr_snapshots
  << "\nperiod_snapshots=" << d.period_snapshots
  << std::endl;
}

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
//const Tensor M_rep = M.unsqueeze(0).unsqueeze(0).repeat({L_stat,L_stat,1,1});

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

const Tensor B = torch::tensor(
  {-4.0/27.0,
    2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0,
    5.0/108.0, 5.0/108.0, 5.0/108.0, 5.0/108.0
  },
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

const Tensor W = torch::tensor(
  {4.0/ 9.0,
    1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

const Tensor E = torch::tensor(
  {{0.0, 1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
   {0.0, 0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0}},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

const Tensor unit_E = E/torch::tensor(
  {1.0, 1.0, 1.0, 1.0, 1.0, std::sqrt(2), std::sqrt(2), std::sqrt(2), std::sqrt(2)},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

double sigmoid(double x) { return 1.0/(1.0 + std::exp(-x)); }

Tensor init_rho_droplet(int R, int C, double rho_0, bool invert)
{
  Tensor rho = torch::zeros({R,C,1},
  torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));
  int rows = rho.size(0);
  int cols = rho.size(1);
  double center = R/2.0;
  double radius = 25.0;

  for(int r=0; r<rows; r++)
  {
    for(int c=0; c<cols; c++)
    {
      double s = std::sqrt((r-center)*(r-center) + (c-center)*(c-center));
      double ans = 0.0;
      if(invert) ans = 1.0 - sigmoid(1.0*(s-radius));
      else ans = sigmoid(1.0*(s-radius));
      rho[r][c] = rho_0*ans;
    }
  }
  return rho.detach().clone();
}

Tensor init_rho_cosine(int R, int C, double rho_0, bool invert)
{
  torch::Tensor rho = torch::zeros({R,C}, torch::kCUDA);
  const int rows = rho.size(0);
  const int cols = rho.size(1);
  const double middle = R/2.0;
  //const double h = 9.0;

  for(int r=0; r<rows; r++)
  {
    for(int c=0; c<cols; c++)
    {
      double s = middle + 0.1*C*std::cos(2.0*3.141592*c/C);
      double ans = 0.0;
      if(invert)
      {
        // Fill red fluid
        if (r>=s) ans = 1.0;
      }
      else
      {
        // Fill blue fluid
        if (r<s) ans = 1.0;
      }
      rho[r][c] = rho_0*ans;
    }
  }
  return rho.unsqueeze(-1).detach().clone();
}

Tensor init_rho(int R, int C, double rho_0, bool invert)
{
  torch::Tensor rho = torch::zeros({R,C}, torch::kCUDA);
  const int rows = rho.size(0);
  const int cols = rho.size(1);
  const double middle = R/2.0;
  //const double h = 9.0;

  for(int r=0; r<rows; r++)
  {
    for(int c=0; c<cols; c++)
    {
      double s = middle;// + 0.1*C*std::cos(2.0*3.141592*c/C);
      double ans = 0.0;
      if(invert)
      {
        // Fill red fluid
        if (r>=s) ans = 1.0;
      }
      else
      {
        // Fill blue fluid
        if (r<s) ans = 1.0;
      }
      rho[r][c] = rho_0*ans;
    }
  }
  return rho.unsqueeze(-1).detach().clone();
}
void eval_phase_field
(
  Tensor& psi,
  double r_rho_0,
  const Tensor& r_rho,
  double b_rho_0,
  const Tensor& b_rho
)
{
  psi.copy_(
    ( r_rho/r_rho_0 - b_rho/b_rho_0 )
    /( r_rho/r_rho_0 + b_rho/b_rho_0 )
  );
}

void update_S(Tensor& S, const Tensor &s_nu)
{
  S.index({Ellipsis, 7, 7}) = s_nu.squeeze(-1).detach().clone();
  S.index({Ellipsis, 8, 8}) = s_nu.squeeze(-1).detach().clone();
}

void eval_equilibrium
(
  Tensor& equ_f,
  const Tensor& k_rho,
  const Tensor& k_phi,
  const Tensor& k_eta,
  const Tensor& u
)
{
  equ_f.copy_(
    k_rho*(
      k_phi + W.mul( 3.0*u.matmul(E)*k_eta + 9.0*u.matmul(E).pow(2) - 3.0*u.mul(u).sum(-1).unsqueeze(-1) )
    )
  );
}

void eval_mrt_operator
(
  Tensor& omega1,
  const Tensor& fk,
  const Tensor& equ_fk,
  const Tensor& Ck,
  const Tensor& S
)
{
  omega1.copy_(
    Mi.matmul( S.matmul( M.matmul( (equ_fk - fk).unsqueeze(-1) ) ) + Ck.unsqueeze(-1) ).squeeze(-1)
  );
}

void eval_per_operator
(
  Tensor& omega2,
  const Tensor& Ak,
  const Tensor& xi
)
{
  omega2.copy_(
    Ak*xi
  );
}

void eval_rec_operator
(
  Tensor& omega3,
  const Tensor& f,
  const Tensor& rhok,
  const Tensor& rho,
  double betak,
  const Tensor& kappa
)
{
  omega3.copy_(
    rhok*f/rho + betak*kappa
  );
}

void eval_xi
(
  Tensor& xi,
  const Tensor& grad,
  const Tensor& grad_norm
)
{
  xi.copy_(
    0.5*grad_norm*( W.mul( ( grad.matmul(E)/(1e-20+grad_norm) ).pow(2) )  - B )
  );
}

void eval_kappa
(
  Tensor& kappa,
  const Tensor& r_rho,
  const Tensor& b_rho,
  const Tensor& rho,
  const Tensor& grad,
  const Tensor& grad_norm,
  const Tensor& r_phi,
  const Tensor& b_phi
)
{
  kappa.copy_(
    ( r_rho*b_rho*grad.matmul(unit_E)*( r_rho*r_phi + b_rho*b_phi ) )
    /(rho.pow(2)*(1e-20+grad_norm))
  );
}

void update_C
(
  Tensor& C,
  const colour& k,
  const Tensor& u,
  const Tensor& s_nu
)
{
  Tensor DxQx = D.x((1.8*k.alpha - 0.8)*k.rho.squeeze(-1)*u.index({Ellipsis,0}));
  Tensor DyQy = D.y((1.8*k.alpha - 0.8)*k.rho.squeeze(-1)*u.index({Ellipsis,1}));
  C.index_put_({Ellipsis,1},
               3.0*(1.0-0.5*1.25)*(DxQx + DyQy)
  );
  C.index_put_({Ellipsis,7},
               (1.0-0.5*s_nu)*(DxQx - DyQy)
  );
}

void apply_boundary_conditions
(
  torch::Tensor &adv_f,
  const torch::Tensor &col_f
);

int main(int argc, char* argv[])
{
  torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(torch::kDouble));
  if (!torch::cuda::is_available())
  {
    cerr << "CUDA is NOT available\n";
  }

  toml::table tbl;
  try{ tbl = toml::parse_file(argv[1]); }
  catch (const toml::parse_error& err)
  {
    cerr << "Parsing failed:\n" << err << "\n";
    return 1;
  }

  const domain domain{tbl["domain"]};
  cout << domain << endl;
  colour r{tbl["red"], domain.R, domain.C, E};
  cout << "RED" << r << endl;
  colour b{tbl["blue"], domain.R, domain.C, E};
  cout << "BLUE" << b << endl;

  // Init. densities
  r.rho.copy_(init_rho_droplet(domain.R, domain.C, r.rho_0, true));
  b.rho.copy_(init_rho_droplet(domain.R, domain.C, b.rho_0, false));

  relaxation_function relax_func{r, b, 0.1};

  Tensor total_f = torch::zeros({domain.R,domain.C,9}, torch::kCUDA);
  Tensor rho = torch::zeros({domain.R,domain.C,1}, torch::kCUDA);
  Tensor u   = torch::zeros({domain.R,domain.C,2}, torch::kCUDA);
  Tensor phase_field = torch::zeros({domain.R,domain.C,1}, torch::kCUDA);
  Tensor grad = torch::zeros({domain.R,domain.C,2}, torch::kCUDA);
  Tensor grad_norm = torch::zeros({domain.R,domain.C,1}, torch::kCUDA);
  Tensor s_nu = torch::zeros({domain.R,domain.C}, torch::kCUDA);
  Tensor temp_s = torch::diagflat(torch::tensor(
    {{0.0, 1.25, 1.14, 0.0, 1.6, 0.0, 1.6, 0.0, 0.0}},
    torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA)));
  Tensor S = temp_s.unsqueeze(0).unsqueeze(0).repeat({domain.R, domain.C, 1, 1});
  Tensor xi = torch::zeros({domain.R,domain.C,9}, torch::kCUDA);
  Tensor kappa = torch::zeros({domain.R,domain.C,9}, torch::kCUDA);
  Tensor A = torch::zeros({domain.R,domain.C,1}, torch::kCUDA);
  const double sigma = 0.1;

  Tensor rhos = torch::zeros({domain.R,domain.C,domain.nr_snapshots});
  Tensor uxs  = torch::zeros({domain.R,domain.C,domain.nr_snapshots});
  Tensor uys  = torch::zeros({domain.R,domain.C,domain.nr_snapshots});
  Tensor s_nus= torch::zeros({domain.R,domain.C,domain.nr_snapshots});
  Tensor phases = torch::zeros({domain.R,domain.C,domain.nr_snapshots});
  Tensor gradx  = torch::zeros({domain.R,domain.C,domain.nr_snapshots});
  Tensor grady  = torch::zeros({domain.R,domain.C,domain.nr_snapshots});
  // Tensor omega1 = torch::zeros({domain.R,domain.C,9,domain.nr_snapshots});
  // Tensor omega2 = torch::zeros({domain.R,domain.C,9,domain.nr_snapshots});
  // Tensor omega3 = torch::zeros({domain.R,domain.C,9,domain.nr_snapshots});

  const Tensor Fg = torch::tensor({{0.0},{-6.25e-6}}, torch::kCUDA);
  Tensor force_source = torch::zeros({domain.R, domain.C, 9}, torch::kCUDA);
  const double ics2 = 3.0;
  const double ics4 = 9.0;
  rho.copy_(r.rho + b.rho);
  u = (u + 0.5*Fg.t()/rho).detach().clone();
  eval_equilibrium(r.adv_f, r.rho, r.phi, r.eta, u);
  eval_equilibrium(b.adv_f, b.rho, b.phi, b.eta, u);

  cout << "main loop" << endl;
  cout << torch::tensor({0}) << endl;
  for (int t=0; t < domain.T; t++)
  {

    if (t%domain.period_snapshots==0)
    {
      const int t_ = t/domain.period_snapshots;
      rhos.index_put_({Ellipsis,t_}, rho.squeeze(-1));
      uxs.index_put_({Ellipsis,t_}, u.index({Ellipsis,0}));
      uys.index_put_({Ellipsis,t_}, u.index({Ellipsis,1}));
      s_nus.index_put_({Ellipsis,t_}, s_nu.squeeze(-1));
      phases.index_put_({Ellipsis,t_}, phase_field.squeeze(-1));
      // omega1.index_put_({Ellipsis,t_}, r.omega1);
      // omega2.index_put_({Ellipsis,t_}, r.omega2);
      // omega3.index_put_({Ellipsis,t_}, r.omega3);
      gradx.index_put_({Ellipsis,t_}, grad.index({Ellipsis,0}));
      grady.index_put_({Ellipsis,t_}, grad.index({Ellipsis,1}));
    }

    eval_equilibrium(r.equ_f, r.rho, r.phi, r.eta, u);
    eval_equilibrium(b.equ_f, b.rho, b.phi, b.eta, u);

    eval_phase_field(phase_field, r.rho_0, r.rho, b.rho_0, b.rho);
    relax_func.eval(s_nu, phase_field);
    update_C(r.C, r, u, s_nu);
    update_C(b.C, b, u, s_nu);
    update_S(S, s_nu);

    eval_mrt_operator(r.omega1, r.adv_f, r.equ_f, r.C, S);
    eval_mrt_operator(b.omega1, b.adv_f, b.equ_f, b.C, S);

    D.grad(grad, phase_field);
    grad_norm.copy_(
      // tnnf::normalize(grad, tnnf::NormalizeFuncOptions().p(2).dim(0))
      torch::sqrt(grad.index({Ellipsis,0}).pow(2) + grad.index({Ellipsis,1}).pow(2)).unsqueeze(-1)
    );
    //grad.masked_fill_(grad_norm<=1e-1*grad_norm.max(), 0.0);
    eval_xi(xi, grad, grad_norm);
    A.copy_( 4.5*sigma*s_nu.unsqueeze(-1) );
    eval_per_operator(r.omega2, A, xi);
    eval_per_operator(b.omega2, A, xi);

    eval_kappa(kappa, r.rho, b.rho, rho, grad, grad_norm, r.phi, b.phi);
    total_f.copy_(r.adv_f + r.omega1 + r.omega2 + b.adv_f + b.omega1 + b.omega2);
    eval_rec_operator(r.omega3, total_f, r.rho, rho, r.beta, kappa);
    eval_rec_operator(b.omega3, total_f, b.rho, rho, b.beta, kappa);

    // Force source terms
    force_source.copy_(
      (1-0.5*s_nu.unsqueeze(-1))*((ics2 + ics4*u.matmul(E))*Fg.t().matmul(E) - ics2*u.matmul(Fg))*W
    );
    r.col_f.copy_(r.omega3 /*+ force_source*/);
    b.col_f.copy_(b.omega3 /*+ force_source*/);

    solver::advect(r.adv_f, r.col_f);
    solver::advect(b.adv_f, b.col_f);

    apply_boundary_conditions(r.adv_f, r.col_f);
    apply_boundary_conditions(b.adv_f, b.col_f);

    r.rho.copy_(r.adv_f.sum(-1).unsqueeze(-1));
    b.rho.copy_(b.adv_f.sum(-1).unsqueeze(-1));
    rho.copy_(r.rho+b.rho);

    solver::calc_u(u, r.adv_f+b.adv_f, rho);
    u = (u + 0.5*Fg.t()/rho).detach().clone();
  }

  cout << "save snapshots" << endl;
  torch::save(rhos, "mrtcg-static-droplet-rhos.pt");
  torch::save(uxs, "mrtcg-static-droplet-uxs.pt");
  torch::save(uys, "mrtcg-static-droplet-uys.pt");
  torch::save(s_nus, "mrtcg-static-droplet-snus.pt");
  torch::save(phases, "mrtcg-static-droplet-phases.pt");
  torch::save(gradx, "mrtcg-static-droplet-gradx.pt");
  torch::save(grady, "mrtcg-static-droplet-grady.pt");
  // torch::save(omega1, "mrtcg-static-droplet-omega1.pt");
  // torch::save(omega2, "mrtcg-static-droplet-omega2.pt");
  // torch::save(omega3, "mrtcg-static-droplet-omega3.pt");

  return 0;
}

void apply_boundary_conditions
(
  torch::Tensor &adv_f,
  const torch::Tensor &col_f
)
{
  // Complete the post-advection population using the post-collision populations
  // adv_f.index(left) = col_f.index(right).detach().clone();
  // adv_f.index(right) = col_f.index(left).detach().clone();
  // adv_f.index(top) = col_f.index(bottom).clone().detach();
  // adv_f.index(bottom) = col_f.index(top).clone().detach();

  // left and right
  // adv_f.index({Slice(1,-1), -1, 3}) = col_f.index({Slice(1,-1), -1, 1}).clone().detach();
  // adv_f.index({Slice(1,-1), -1, 7}) = col_f.index({Slice(1,-1), -1, 5}).clone().detach();
  // adv_f.index({Slice(1,-1), -1, 6}) = col_f.index({Slice(1,-1), -1, 8}).clone().detach();

  // adv_f.index({Slice(1,-1),  0, 1}) = col_f.index({Slice(1,-1), 0, 3}).clone().detach();
  // adv_f.index({Slice(1,-1),  0, 5}) = col_f.index({Slice(1,-1), 0, 7}).clone().detach();
  // adv_f.index({Slice(1,-1),  0, 8}) = col_f.index({Slice(1,-1), 0, 6}).clone().detach();

  // inlet-oulet
  adv_f.index({Slice(1,-1), 0, 2}) = col_f.index({Slice(1, -1),-1, 2});
  adv_f.index({Slice(1,-1), 0, 5}) = col_f.index({Slice(1, -1),-1, 5});
  adv_f.index({Slice(1,-1), 0, 6}) = col_f.index({Slice(1, -1),-1, 6});

  adv_f.index({Slice(1,-1),-1, 4}) = col_f.index({Slice(1, -1), 0, 4});
  adv_f.index({Slice(1,-1),-1, 8}) = col_f.index({Slice(1, -1), 0, 8});
  adv_f.index({Slice(1,-1),-1, 7}) = col_f.index({Slice(1, -1), 0, 7});

  adv_f.index({-1, Slice(), 3}) = col_f.index({-1, Slice(), 1}).detach().clone();
  adv_f.index({-1, Slice(), 7}) = col_f.index({-1, Slice(), 5}).detach().clone();
  adv_f.index({-1, Slice(), 6}) = col_f.index({-1, Slice(), 8}).detach().clone();

  adv_f.index({ 0, Slice(), 1}) = col_f.index({ 0, Slice(), 3}).detach().clone();
  adv_f.index({ 0, Slice(), 5}) = col_f.index({ 0, Slice(), 7}).detach().clone();
  adv_f.index({ 0, Slice(), 8}) = col_f.index({ 0, Slice(), 6}).detach().clone();

}
