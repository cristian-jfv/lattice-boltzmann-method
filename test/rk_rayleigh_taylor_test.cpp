#include <ATen/TensorIndexing.h>
#include <cmath>
#include <iostream>
#include <torch/serialize.h>

#include "../src/rk.hpp"
#include "../src/solver.hpp"
#include "../src/utils.hpp"

using std::cout;
using std::endl;

using torch::indexing::Ellipsis;
using torch::indexing::Slice;

const utils::indices top{0,Ellipsis};
const utils::indices v_top{1,Ellipsis};

const utils::indices bottom{-1,Ellipsis};
const utils::indices v_bottom{-2,Ellipsis};

const utils::indices left{Slice(1, -1), 0, Ellipsis};
const utils::indices v_left{Slice(1, -1), 1, Ellipsis};

const utils::indices right{Slice(1, -1), -1, Ellipsis};
const utils::indices v_right{Slice(1, -1), -2, Ellipsis};

torch::Tensor init_rho
(
  double rho_0,
  int R,
  int C,
  bool invert
);

void apply_boundary_conditions
(
  torch::Tensor &adv_f,
  const torch::Tensor &col_f
)
{
  // Complete the post-advection population using the post-collision populations
  adv_f.index(left) = col_f.index(right).detach().clone();
  adv_f.index(right) = col_f.index(left).detach().clone();
  // adv_f.index(top) = col_f.index(bottom).clone().detach();
  // adv_f.index(bottom) = col_f.index(top).clone().detach();

  adv_f.index({-1, Slice(), 4}) = col_f.index({-1, Slice(), 2}).detach().clone();
  adv_f.index({-1, Slice(), 7}) = col_f.index({-1, Slice(), 5}).detach().clone();
  adv_f.index({-1, Slice(), 8}) = col_f.index({-1, Slice(), 6}).detach().clone();

  adv_f.index({0, Slice(), 2}) = col_f.index({0, Slice(), 4}).detach().clone();
  adv_f.index({0, Slice(), 5}) = col_f.index({0, Slice(), 7}).detach().clone();
  adv_f.index({0, Slice(), 6}) = col_f.index({0, Slice(), 8}).detach().clone();
}

int main()
{
  using torch::Tensor;
  const auto dev = torch::kCUDA;
  torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(torch::kDouble));

  const int T = 10000; // time steps
  const int L_ = 256; // domain size
  const int R = 4*L_;
  const int C = L_;

  const double sigma=1e-1;
  const double nu=0.04;
  // 1.- Initialization of densities and velocities
  colour r{R,C,/*alpha=*/11.0/15.0, /*rho_0=*/3.0, /*nu=*/nu, /*beta=*/-0.7,
    init_rho(3.0, R, C, true)};
  // betas for both colours should be equal
  // here the definitions use opposite signs to be able to use the same kappa tensor
  // for the recoloring operation
  colour b{R,C,/*alpha=*/0.2, /*rho_0=*/1.0, /*nu=*/nu, /*beta=*/0.7,
    init_rho(1.0, R, C, false)};
  Tensor u =   torch::zeros({R,C,2}, dev);
  Tensor rho = torch::zeros({R,C,1}, dev);
  rho.copy_(r.rho+b.rho);

  rk rk
  {
    R, C,
    r.alpha, r.rho_0, r.nu, r.rho,
    b.alpha, b.rho_0, b.nu, b.rho,
    sigma, u
  };
  Tensor omega1 = torch::zeros({R,C,9}, dev);
  Tensor omega2 = torch::zeros({R,C,9}, dev);
  Tensor omega3 = torch::zeros({R,C,9}, dev);
  Tensor f =      torch::zeros({R,C,9}, dev); // colour-blind dist. function
  Tensor adv_f  = torch::zeros({R,C,9}, dev);

  const int snaps = 1000;
  const int div = 10;
  Tensor r_rhos = torch::zeros({R,C,snaps});
  Tensor b_rhos = torch::zeros({R,C,snaps});
  Tensor uxs  =   torch::zeros({R,C,snaps});
  Tensor uys  =   torch::zeros({R,C,snaps});

  // 2.- Initialize the colour-blind dist. function
  rk.eval_equilibrium(f, r.rho, b.rho, rho, u);

  // Gravity
  const Tensor Fg = torch::tensor({{-6.25e-6},{0.0}}, dev);
  Tensor S = torch::zeros({R,C,9}, dev); // force source term
  const double ics2 = 3.0;
  const double ics4 = 9.0;

  // 3.- Main loop
  cout << "main loop" << endl;
  for (int t=0; t<T; t++)
  {
    // Snapshot
    if (t%div == 0)
    {
      const int a = (int)(t/div);
      r_rhos.index({Ellipsis,a}) = r.rho.squeeze(-1).detach().clone();
      b_rhos.index({Ellipsis,a}) = b.rho.squeeze(-1).detach().clone();
      uxs.index({Ellipsis,a}) = u.index({Ellipsis,0}).detach().clone();
      uys.index({Ellipsis,a}) = u.index({Ellipsis,1}).detach().clone();
    }

    rk.update_aux_quantities(r.rho, b.rho, rho, u);
    // 3.1.- BGK collision
    rk.eval_bgk(omega1, f, r.rho, b.rho, rho);
    S.copy_(
      (1-0.5*rk.get_omega_rp())*((ics2 + ics4*u.matmul(rk.E))*Fg.t().matmul(rk.E) - ics2*u.matmul(Fg))*rk.W
    );
    // 3.2.- Compute gradients
    // gradients are already computed

    // 3.3.- Perturbation step
    rk.eval_perturbation(omega2, r.rho, b.rho);
    omega2.add_(omega1 + S);

    // 3.4.- temp_f=0, temp_rho=0
    // Use copy operation to skip this step and replace values directly

    // 3.5.RED.- recolouring
    rk.eval_recolouring(omega3, omega2, r.rho, rho, r.beta);
    solver::advect(adv_f, omega3);
    apply_boundary_conditions(adv_f, omega3);
    r.rho.copy_(adv_f.sum(-1).unsqueeze(-1));
    f.copy_(adv_f);

    // 3.5.BLUE.- recolouring
    rk.eval_recolouring(omega3, omega2, b.rho, rho, b.beta);
    solver::advect(adv_f, omega3);
    apply_boundary_conditions(adv_f, omega3);
    b.rho.copy_(adv_f.sum(-1).unsqueeze(-1));
    f.add_(adv_f);

    rho.copy_(r.rho + b.rho);
    solver::calc_u(u, f, rho);
  }

  cout << "save results" << endl;
  torch::save(r_rhos, "rk-rayleigh-taylor-r-rhos.pt");
  torch::save(b_rhos, "rk-rayleigh-taylor-b-rhos.pt");
  torch::save(uxs, "rk-rayleigh-taylor-uxs.pt");
  torch::save(uys, "rk-rayleigh-taylor-uys.pt");

  return 0;
}

double sigmoid(double x) { return 1.0/(1.0 + std::exp(-x)); }

torch::Tensor init_rho(double rho_0, const int R, const int C, bool invert)
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
