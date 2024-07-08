#include "rk.hpp"
#include "utils.hpp"
#include <ATen/TensorIndexing.h>

void rk::eval_recolouring
(
  torch::Tensor& omega3_,
  const torch::Tensor& f,
  const torch::Tensor& k_rho,
  const torch::Tensor& rho,
  double beta
)
{
  omega3_.copy_(
    k_rho*f/rho + beta*kappa
  );
}

void rk::eval_perturbation
(
  torch::Tensor& omega2_,
  const torch::Tensor& r_rho,
  const torch::Tensor& b_rho
)
{
  E_u.copy_(torch::matmul(F_kl,E).pow(2.0));
  omega2_.copy_(
    A_kl*C_kl*norm_F_kl*( torch::mul(E_u/(1e-20+norm_F_kl.pow(2.0)), W) - B )
  );
}

void rk::eval_bgk
(
  torch::Tensor &omega1_,
  const torch::Tensor& f,
  const torch::Tensor& r_rho,
  const torch::Tensor& b_rho,
  const torch::Tensor& rho
)
{
  omega1_.copy_(
    (1-omega_rp)*f + omega_rp*equ_f
  );
}

void rk::eval_equilibrium
(
  torch::Tensor& equ_f,
  const torch::Tensor& r_rho,
  const torch::Tensor& b_rho,
  const torch::Tensor& rho,
  const torch::Tensor& u
)
{
  E_u.copy_(torch::matmul(u, E));
  // utils::print((u*u).sum(-1).sizes());
  u_u.copy_((u*u).sum(-1).unsqueeze(-1));
  eval_phi(r_rho, b_rho, rho);

  equ_f.copy_(
    rho*( phi + torch::mul( 3.0*E_u + 4.5*E_u.pow(2.0) - 1.5*u_u , W) )
  );
}

void rk::update_aux_quantities
(
  const torch::Tensor& r_rho,
  const torch::Tensor& b_rho,
  const torch::Tensor& rho,
  const torch::Tensor& u
)
{
  eval_omega_rp(r_rho, b_rho, rho);
  eval_alpha(r_rho, b_rho, rho);
  eval_colour_gradient(r_rho, b_rho, rho);
  // eval only required for equilibrium eval_phi(r_rho, b_rho, rho);
  eval_C_kl(r_rho, b_rho);
  eval_A_kl(omega_rp);
  eval_equilibrium(equ_f, r_rho, b_rho, rho, u);
}

void rk::eval_kappa
(
  const torch::Tensor& r_rho,
  const torch::Tensor& b_rho,
  const torch::Tensor& rho
)
{
  kappa.copy_(
    r_rho*b_rho*torch::matmul(F_kl/(norm_F_kl+1e-20), unit_E)*phi/rho
  );
}

void rk::eval_colour_gradient
(
  const torch::Tensor& r_rho,
  const torch::Tensor& b_rho,
  const torch::Tensor& rho
)
{
  using torch::indexing::Ellipsis;
  F_kl.zero_();
  D.grad(temp_F_kl, r_rho/rho);
  F_kl.copy_(b_rho*temp_F_kl/rho);

  D.grad(temp_F_kl, b_rho/rho);
  F_kl.add_(-r_rho*temp_F_kl/rho);

  // torch::Tensor temp = torch::sqrt(
  //     F_kl.index({Ellipsis,0}).pow(2.0) + F_kl.index({Ellipsis,1}).pow(2.0)
  //   );
  // utils::print(temp.sizes());

  norm_F_kl.copy_(
    torch::sqrt(
      F_kl.index({Ellipsis,0}).pow(2.0) + F_kl.index({Ellipsis,1}).pow(2.0)
    ).unsqueeze(-1)
  );

  // Normalize gradient
  //utils::print(norm_F_kl.sizes());
  // norm_F_kl.masked_fill_(norm_F_kl < 1e-2*norm_F_kl.max(), 0.0);
  // F_kl.masked_fill_(norm_F_kl < 1e-2*norm_F_kl.max(), 0.0);
  // F_kl.div_(norm_F_kl + 1e-20);
}

void rk::eval_alpha
(
    const torch::Tensor& r_rho,
    const torch::Tensor& b_rho,
    const torch::Tensor& rho
)
{
  alpha.copy_(
    ( r_alpha*r_rho + b_alpha*b_rho )/rho
  );
}

void rk::eval_phi
(
    const torch::Tensor& r_rho,
    const torch::Tensor& b_rho,
    const torch::Tensor& rho
)
{
  using torch::indexing::Slice;
  using torch::indexing::None;
  // TODO: check dimensions after copy, and values after put
  // utils::print((0.2-0.2*alpha).repeat({1,1,4}).sizes());

  phi.index_put_({Slice(),Slice(),0}, alpha.squeeze(-1));
  phi.index_put_({Slice(),Slice(),Slice(1,5)}, (0.2-0.2*alpha).repeat({1,1,4}));
  phi.index_put_({Slice(),Slice(),Slice(5,None)}, (0.05-0.05*alpha).repeat({1,1,4}));
}

void rk::eval_A_kl
(
  const torch::Tensor& omega_rp
)
{
  A_kl.copy_(
    4.5*sigma*omega_rp
  );
}

void rk::eval_C_kl
(
  const torch::Tensor& r_rho,
  const torch::Tensor& b_rho
)
{
/*
  C_kl.copy_(
    ( eta/(r_rho_0*b_rho_0) )*r_rho*b_rho
  );
  C_kl.masked_fill_(C_kl>1.0, 1.0);
*/
  C_kl.copy_(
    1.0 - torch::abs( ( r_rho - b_rho ) / ( r_rho + b_rho ) )
  );
  // C_kl.masked_fill_(C_kl < 0.5, 0.0);
}

void rk::eval_omega_rp
(
  const torch::Tensor& r_rho,
  const torch::Tensor& b_rho,
  const torch::Tensor& rho
)
{
  omega_rp.copy_(
    rho/( 3.0*r_rho*r_nu + 3.0*b_rho*b_nu + 0.5*rho )
  );
}

rk::rk
(
  int R,
  int C,
  double r_alpha,
  double r_rho_0,
  double r_nu,
  torch::Tensor& r_rho,
  double b_alpha,
  double b_rho_0,
  double b_nu,
  torch::Tensor& b_rho,
  double sigma,
  torch::Tensor u
):
sigma{sigma},
r_rho_0{r_rho_0},
r_alpha{r_alpha},
r_nu{r_nu},
b_rho_0{b_rho_0},
b_alpha{b_alpha},
b_nu{b_nu}
{
  auto dev = torch::kCUDA;

  // Give tensors initial sizes
  phi    = torch::zeros({R,C,9}, dev);

  E_u = torch::zeros({R,C,9}, dev);
  u_u = torch::zeros({R,C,1}, dev);

  alpha = torch::zeros({R,C,1}, dev);
  A_kl = torch::zeros({R,C,1}, dev);
  C_kl = torch::zeros({R,C,1}, dev);
  F_kl = torch::zeros({R,C,2}, dev);
  norm_F_kl = torch::zeros({R,C,1}, dev);
  temp_F_kl = torch::zeros({R,C,2}, dev);
  omega_rp = torch::zeros({R,C,1}, dev);
  equ_f  = torch::zeros({R,C,9}, dev);
  kappa = torch::zeros({R,C,9}, dev);

  update_aux_quantities(r_rho, b_rho, r_rho+b_rho, u);
}

colour::colour(int R, int C, double alpha, double rho_0, double nu, double beta,
               const torch::Tensor& init_rho):
alpha{alpha},
rho_0{rho_0},
nu{nu},
beta{beta}
{
  rho = torch::zeros({R,C,1}, torch::kCUDA);
  rho.copy_(init_rho);
}


