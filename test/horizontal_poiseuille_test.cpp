#include <iostream>
#include <cmath>
#include <ostream>
#include <cassert>
#include <torch/torch.h>
#include <torch/types.h>

#include "../src/solver.hpp"
#include "../src/utils.hpp"

using std::cout;
using std::endl;
using torch::Tensor;
using torch::indexing::Slice;
using torch::indexing::Ellipsis;
using utils::print;
using utils::indices;

// Boundaries
const indices virtual_inlet{0, Ellipsis};
const indices inlet{1, Ellipsis};
const indices outlet{-2, Ellipsis};
const indices virtual_outlet{-1, Ellipsis};

void periodic_boundary_condition
(
  torch::Tensor& f_coll,
  const torch::Tensor& f_equi,
  const torch::Tensor& u,
  const torch::Tensor& rho,
  const double rho_inlet,
  const double rho_outlet
)
{
  torch::Tensor temp_equi = torch::zeros({1, u.sizes()[1], 9}, torch::kCUDA);
  torch::Tensor temp_rho = torch::ones({1, u.sizes()[1], 1}, torch::kCUDA);

  // inlet
  solver::incomp_equilibrium(temp_equi, u.index(outlet).unsqueeze(0), rho_inlet*temp_rho);
  f_coll.index(virtual_inlet) = (temp_equi + f_coll.index(outlet) - f_equi.index(outlet)).squeeze(0).clone().detach();

  // outlet
  solver::incomp_equilibrium(temp_equi, u.index(inlet).unsqueeze(0), rho_outlet*temp_rho);
  f_coll.index(virtual_outlet) = (temp_equi + f_coll.index(inlet) - f_equi.index(inlet)).squeeze(0).clone().detach();
}

int main()
{
  // Flow parameters
  const int T = 8301;
  print("T", T);
  const int H = 21;
  const int W = 21;
  cout << "H=" << H << "; W=" << W << endl;
  const double tau = std::sqrt(3.0/16.0) + 0.5;
  const double omega = 1.0/tau;
  cout << "omega=" << omega << endl;
  const double u_max = 1.030985714E-1; //0.1;
  const double nu = (2.0 * tau -1.0)/6.0;
  cout << "nu=" << nu << endl;
  const double Re = W*u_max/nu;
  cout << "Re=" << Re << endl;
  const double p_grad = 8.0*nu*u_max/(W*W);
  cout << "grad(p)=" << p_grad << endl;
  const double rho_outlet = 1.0;
  const double rho_inlet = 3.0*(H-1)*p_grad + rho_outlet;
  cout << "rho_inlet=" << rho_inlet << endl;

  torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(torch::kDouble));
  cout << "Default torch dtype: " << torch::get_default_dtype() << endl;
  if (!torch::cuda::is_available())
  {
    std::cerr << "CUDA is NOT available\n";
  }

  const torch::Device dev = torch::kCUDA;
  // Tensors
  Tensor f_equi = torch::zeros({H,W,9}, dev);
  Tensor f_coll = torch::zeros_like(f_equi, dev);
  Tensor f_adve = torch::zeros_like(f_equi, dev);
  Tensor u = torch::zeros({H,W,2}, dev);
  Tensor rho = torch::ones({H,W,1}, dev);

  // Results
  Tensor fs = torch::zeros({H,W,9,T});
  Tensor ux = torch::zeros({H,W,T});
  Tensor uy = torch::zeros({H,W,T});
  Tensor rhos = torch::zeros({H,W,T});

  // Initialisation
  solver::incomp_equilibrium(f_adve, u, rho);

  // Convergence test
  const int t_interval = 100;
  const double tolerance = 1e-12;
  Tensor old_u = torch::ones_like(rho);

  // Main loop
  print("main loop starts");
  for (int t=0; t < T; t++)
  {
    cout << t << "\t\r" << std::flush;
    //print("save fs");
    fs.index_put_({Ellipsis,t}, f_adve);
    //print("save ux");
    ux.index_put_({Ellipsis,t}, u.index({Ellipsis,0}));
    //print("save uy");
    uy.index_put_({Ellipsis,t}, u.index({Ellipsis,1}));
    //print("save rho");
    rhos.index_put_({Ellipsis,t}, rho.squeeze(2));

    // Check for convergence
    if (t % t_interval == 1)
    {
      double diff = torch::abs( u.index({Ellipsis,0}).mean()/old_u.index({Ellipsis,0}).mean() - 1.0 ).item<double>();

      if (diff < tolerance)
      {
        print("last t", t);
        break;
      }
      else
      {
        old_u = u.clone().detach();
      }
    }

    // Calculate macroscopic variables
    //print("calculate macroscopic variables");
    solver::calc_rho(rho, f_adve);
    solver::calc_incomp_u(u, f_adve);
    // Compute equilibrium
    //print("compute equilibrium");
    solver::incomp_equilibrium(f_equi, u, rho);
    // Collision, BGK operator
    //print("collision step");
    solver::collision(f_coll, f_adve, f_equi, omega);
    // Inlet and outlet periodic boundary conditions
    //print("inlet-outlet priodic boundary condition");
    periodic_boundary_condition(f_coll, f_equi, u, rho, rho_inlet, rho_outlet);
    // Advection
    //print("advection step");
    solver::advect(f_adve, f_coll);

    // No slip boundary conditions for the first and last column
    f_adve.index({Slice(), -1, 4}) = f_coll.index({Slice(), -1, 2}).clone().detach();
    f_adve.index({Slice(), -1, 7}) = f_coll.index({Slice(), -1, 5}).clone().detach();
    f_adve.index({Slice(), -1, 8}) = f_coll.index({Slice(), -1, 6}).clone().detach();

    f_adve.index({Slice(), 0, 2}) = f_coll.index({Slice(), 0, 4}).clone().detach();
    f_adve.index({Slice(), 0, 5}) = f_coll.index({Slice(), 0, 7}).clone().detach();
    f_adve.index({Slice(), 0, 6}) = f_coll.index({Slice(), 0, 8}).clone().detach();
  }

  // Save results
  print("saving results into files");
  torch::save(ux, "hpt-ux.pt");
  torch::save(uy, "hpt-uy.pt");
  torch::save(fs, "hpt-fs.pt");
  torch::save(rhos/3.0, "hpt-ps.pt");

  // Evaluate final velocity profile
  Tensor y = torch::linspace(1,W,W) - 0.5;
  Tensor u_analytical = -4.0*u_max/(W*W)*y*(y-W);
  Tensor errors = torch::zeros({H});
  auto denominator = 1.0/torch::sqrt(torch::sum(torch::pow(u_analytical, 2.0)));
  for (int r = 1; r < H-1; r++)
  {
    errors.index({r}) = torch::sqrt( torch::sum( torch::pow( u.index({r,Slice(),0}) - u_analytical , 2.0) ) )*denominator;
  }

  auto l2 = (1.0/H)*torch::sum(errors).item<double>();
  print("L2", l2);

  assert((l2 <= 1e-11) && "Large L2 error");
  return 0;
}
