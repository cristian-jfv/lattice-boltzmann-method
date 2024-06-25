#ifndef RK_HPP
#define RK_HPP

#include <torch/nn/functional/normalization.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/options/normalization.h>
#include <torch/torch.h>

class diff_op
{
  private:
  const torch::Tensor xi = (1.0/5040.0)*torch::tensor(
    {{ 1.0,  32.0,  84.0,  32.0,  1.0},
     {32.0, 448.0, 960.0, 448.0, 32.0},
     {84.0, 960.0,   0.0, 960.0, 84.0},
     {32.0, 448.0, 960.0, 448.0, 32.0},
     { 1.0,  32.0,  84.0,  32.0,  1.0}},
    torch::TensorOptions()
    .dtype(torch::kDouble)
    .device(torch::kCUDA)
    .requires_grad(false));

  const torch::Tensor kernel_partial_x = torch::tensor(
    {{-2.0, -1.0, 0.0, 1.0, 2.0},
     {-2.0, -1.0, 0.0, 1.0, 2.0},
     {-2.0, -1.0, 0.0, 1.0, 2.0},
     {-2.0, -1.0, 0.0, 1.0, 2.0},
     {-2.0, -1.0, 0.0, 1.0, 2.0}},
    torch::TensorOptions()
    .dtype(torch::kDouble)
    .device(torch::kCUDA)
    .requires_grad(false));

  const torch::Tensor kernel_partial_y = -torch::tensor(
    {{ 2.0,  2.0,  2.0,  2.0,  2.0},
     { 1.0,  1.0,  1.0,  1.0,  1.0},
     { 0.0,  0.0,  0.0,  0.0,  0.0},
     {-1.0, -1.0, -1.0, -1.0, -1.0},
     {-2.0, -2.0, -2.0, -2.0, -2.0}},
    torch::TensorOptions()
    .dtype(torch::kDouble)
    .device(torch::kCUDA)
    .requires_grad(false));

  // Convolution operators for partial derivatives
  torch::nn::Conv2d partial_x = nullptr;
  torch::nn::Conv2d partial_y = nullptr;
  torch::nn::Conv2d initialize_convolution(const torch::Tensor &kernel);

  public:
  diff_op();
  torch::Tensor x(const torch::Tensor& psi);
  torch::Tensor y(const torch::Tensor& psi);
  void grad(torch::Tensor& ans, const torch::Tensor& psi);

};

class rk
{
  public:

  torch::Tensor get_omega_rp(){return omega_rp.detach().clone();};

  void eval_equilibrium
  (
    torch::Tensor& equ_f,
    const torch::Tensor& r_rho,
    const torch::Tensor& b_rho,
    const torch::Tensor& rho,
    const torch::Tensor& u
  );

  void eval_bgk
  (
    torch::Tensor& omega1,
    const torch::Tensor& f,
    const torch::Tensor& r_rho,
    const torch::Tensor& b_rho,
    const torch::Tensor& rho
  );

  void eval_perturbation
  (
    torch::Tensor& omega2_,
    const torch::Tensor& r_rho,
    const torch::Tensor& b_rho
  );

  void eval_recolouring
  (
    torch::Tensor& omega3_,
    const torch::Tensor& f,
    const torch::Tensor& k_rho,
    const torch::Tensor& rho,
    double beta
  );

  void update_aux_quantities
  (
    const torch::Tensor& r_rho,
    const torch::Tensor& b_rho,
    const torch::Tensor& rho,
    const torch::Tensor& u
  );

  rk
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
  );
  const torch::Tensor W = torch::tensor(
    {4.0/ 9.0,
      1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0,
      1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0},
    torch::TensorOptions()
    .dtype(torch::kDouble)
    .device(torch::kCUDA)
    .requires_grad(false));

  const torch::Tensor E = torch::tensor(
    {{0.0, 1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
     {0.0, 0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0}},
    torch::TensorOptions()
    .dtype(torch::kDouble)
    .device(torch::kCUDA)
    .requires_grad(false));

  private:

  const torch::Tensor unit_E = torch::nn::functional::normalize(E,
    torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(0));

  const torch::Tensor B = torch::tensor(
  {-4.0/27.0,
    2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0,
    5.0/108.0, 5.0/108.0, 5.0/108.0, 5.0/108.0},
    torch::TensorOptions()
    .dtype(torch::kDouble)
    .device(torch::kCUDA)
    .requires_grad(false));

  const double eta=1e6;
  const double sigma;
  const double r_rho_0;
  const double r_alpha;
  const double r_nu;
  const double b_rho_0;
  const double b_alpha;
  const double b_nu;

  diff_op D;

  // Storage for intermediate results
  // Never assume they contain the correct value
  // Always call the appropiate eval funciton before using them
  torch::Tensor phi;
  torch::Tensor E_u;
  torch::Tensor u_u;
  torch::Tensor alpha;
  torch::Tensor A_kl;
  torch::Tensor C_kl;
  torch::Tensor F_kl;
  torch::Tensor norm_F_kl;
  torch::Tensor temp_F_kl;
  torch::Tensor omega_rp;
  torch::Tensor equ_f;
  torch::Tensor kappa;

  void eval_kappa
  (
    const torch::Tensor& r_rho,
    const torch::Tensor& b_rho,
    const torch::Tensor& rho
  );

  void eval_colour_gradient
  (
    const torch::Tensor& r_rho,
    const torch::Tensor& b_rho,
    const torch::Tensor& rho
  );

  void eval_alpha
  (
    const torch::Tensor& r_rho,
    const torch::Tensor& b_rho,
    const torch::Tensor& rho
  );

  void eval_phi
  (
    const torch::Tensor& r_rho,
    const torch::Tensor& b_rho,
    const torch::Tensor& rho
  );

  void eval_C_kl
  (
    const torch::Tensor& r_rho,
    const torch::Tensor& b_rho
  );

  void eval_A_kl
  (
    const torch::Tensor& omega_rp
  );

  void eval_omega_rp
  (
    const torch::Tensor& r_rho,
    const torch::Tensor& b_rho,
    const torch::Tensor& rho
  );
};

class colour
{
  public:
  const double alpha;
  const double rho_0;
  const double nu;
  const double beta;
  torch::Tensor rho;

  colour(int R, int C, double alpha, double rho_0, double nu, double beta,
         const torch::Tensor& init_rho);
};


#endif
