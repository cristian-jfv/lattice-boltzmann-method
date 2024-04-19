#include <cassert>
#include <iostream>
#include <ostream>
#include <torch/serialize.h>
#include <torch/torch.h>
#include "../src/solver.hpp"
#include "../src/domain.hpp"

using namespace std;

void store_tensor(const torch::Tensor& source, torch::Tensor& target, int i)
{
  using torch::indexing::Slice;
    // Save results
    target.index({Slice(), Slice(), Slice(), i}) =
      source.index({Slice(), Slice(), Slice()}).clone().detach();
}

int main()
{
  using torch::Tensor;
  using torch::indexing::Slice;
  using torch::indexing::None;

  cout << "\nProgram starts" << endl;
  const double rho_0 = 1.0;
  const double p_0 = 1.0;
  const double tau = 0.95;
  const double eps = 1.0/tau;
  const int H = 51;
  const int W = 51;
  const int T = 10000; // TODO: Calculate number of time steps required
  const double dp = 0.00108;

  // Initialize matrices
  Tensor f_curr = torch::zeros({H,W,9});
  Tensor f_next = torch::zeros_like(f_curr);
  Tensor f_equi = torch::zeros_like(f_curr);
  Tensor u_tens = torch::zeros({H,W,2});
  Tensor p_tens = torch::zeros({H,W,1});

  solver::initialize(rho_0, p_0, f_curr, u_tens, p_tens);
  // Create views for the domain boundaries
  auto top = domain::top_boundary(f_next);
  auto bottom = domain::bottom_boundary(f_next);
  auto left = domain::left_boundary(f_next);
  auto right = domain::right_boundary(f_next);

  // Results storage
  Tensor ux = torch::zeros({H, W, T});
  Tensor uy = torch::zeros({H, W, T});
  Tensor ps = torch::zeros({H, W, T});
  Tensor fs = torch::zeros({H, W, 9, T});

  // Main loop
  cout << "===================================================================" << endl;
  cout << "Main loop starts" << endl;
  cout << "===================================================================" << endl;

  for (int i=0; i<T; i++)
  {
    cout << i << "\t\r" << flush;
    // Save results
    ux.index({Slice(), Slice(), i}) = u_tens.index({Slice(), Slice(), 0}).clone().detach();
    uy.index({Slice(), Slice(), i}) = u_tens.index({Slice(), Slice(), 1}).clone().detach();
    ps.index({Slice(), Slice(), i}) = p_tens.index({Slice(), Slice(), 0}).clone().detach();
    store_tensor(f_curr, fs, i);

    solver::f_eq(f_equi, u_tens, p_tens);
    solver::f_step(f_next, f_curr, f_equi, eps);

    // Enforce boundary conditions
    // Inlet at LEFT
    domain::inlet(left, right, dp);
    // Outlet at RIGHT
    domain::outlet(left, right, dp);
    // No slip at TOP
    domain::no_slip(top, domain::interface::fluid_to_wall);
    // No slip at BOTTOM
    domain::no_slip(bottom, domain::interface::wall_to_fluid);

    // Recover coners
    solver::recover_corners(f_next);
/*  f_next.index({0,0,6-1}) = f_next.index({0,0,8-1}).clone().detach();
    f_next.index({0,-1,6-1}) = f_next.index({0,-1,8-1}).clone().detach();

    f_next.index({0,0,7-1}) = f_next.index({0,0,9-1}).clone().detach();
    f_next.index({0,-1,7-1}) = f_next.index({0,-1,9-1}).clone().detach();

    f_next.index({-1,0,8-1}) = f_next.index({-1,0,6-1}).clone().detach();
    f_next.index({-1,-1,8-1}) = f_next.index({-1,-1,6-1}).clone().detach();

    f_next.index({-1,0,9-1}) = f_next.index({-1,0,7-1}).clone().detach();
    f_next.index({-1,-1,9-1}) = f_next.index({-1,-1,7-1}).clone().detach();
*/
    f_curr = f_next.clone().detach();

    solver::u(u_tens, f_curr);
    solver::p(p_tens, f_curr);
  }

  // Save results
  if (true)
  {
    torch::save(ux, "hpt-ux.pt");
    torch::save(uy, "hpt-uy.pt");
    torch::save(ps, "hpt-ps.pt");
    torch::save(fs, "hpt-fs.pt");
  }
  // Check results
  Tensor last_ux = u_tens.index({Slice(), 0, 0}).clone().detach()/0.03;
  Tensor x = torch::linspace(0, 1, W);
  Tensor analytical_u = 6.0*x*(1.0 - x);
  const double rmse = ( ((last_ux - analytical_u).pow(2)).sum()/W  ).sqrt().item<double>();
  const double error_percent = rmse*100.0/1.5;
  cout << "RMSE=" << rmse << endl;
  cout << "percent=" << error_percent << endl;
  /*
  assert(error_percent <= 1.0 
         && "Simulation yields a different result to the analytical solution");
  */
  cout << "Program ends" << endl;
  return 0;
}
