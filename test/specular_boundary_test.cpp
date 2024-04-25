#include <iostream>
#include <cmath>
#include <ostream>
#include <torch/torch.h>

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
  torch::Tensor temp_equi = torch::zeros({1, u.sizes()[1], 9});
  torch::Tensor temp_rho = torch::ones({1, u.sizes()[1], 1});

  // inlet
  solver::equilibrium(temp_equi, u.index(outlet).unsqueeze(0), rho_inlet*temp_rho);
  f_coll.index(virtual_inlet) = (temp_equi + f_coll.index(outlet) - f_equi.index(outlet)).squeeze(0).clone().detach();

  // outlet
  solver::equilibrium(temp_equi, u.index(inlet).unsqueeze(0), rho_outlet*temp_rho);
  f_coll.index(virtual_outlet) = (temp_equi + f_coll.index(inlet) - f_equi.index(inlet)).squeeze(0).clone().detach();
}

int main()
{
  // Flow parameters
  const int T = 10000;
  print("T", T);
  const int H = 51;
  const int W = 51;
  cout << "H=" << H << "; W=" << W << endl;
  const double tau = std::sqrt(3.0/16.0) + 0.5;
  const double omega = 1.0/tau;
  cout << "omega=" << omega << endl;
  const double u_max = 0.1;
  const double nu = (2.0 * tau -1.0)/6.0;
  cout << "nu=" << nu << endl;
  const double Re = W*u_max/nu;
  cout << "Re=" << Re << endl;
  const double p_grad = 8.0*nu*u_max/(W*W);
  cout << "grad(p)=" << p_grad << endl;
  const double rho_outlet = 1.0;
  const double rho_inlet = 3.0*(H-1)*p_grad + rho_outlet;
  cout << "rho_inlet=" << rho_inlet << endl;

  // Tensors
  Tensor f_equi = torch::zeros({H,W,9});
  Tensor f_coll = torch::zeros_like(f_equi);
  Tensor f_adve = torch::zeros_like(f_equi);
  Tensor u = torch::zeros({H,W,2});
  Tensor rho = torch::ones({H,W,1});

  // Results
  Tensor fs = torch::zeros({H,W,9,T});
  Tensor ux = torch::zeros({H,W,T});
  Tensor uy = torch::zeros({H,W,T});
  Tensor rhos = torch::zeros({H,W,T});

  // Initialisation
  solver::equilibrium(f_adve, u, rho);

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

    // Calculate macroscopic variables
    //print("calculate macroscopic variables");
    solver::calc_rho(rho, f_adve);
    solver::calc_u(u, f_adve, rho);
    // Compute equilibrium
    //print("compute equilibrium");
    solver::equilibrium(f_equi, u, rho);
    // Collision, BGK operator
    //print("collision step");
    solver::collision(f_coll, f_adve, f_equi, omega);
    // Inlet and outlet periodic boundary conditions
    //print("inlet-outlet priodic boundary condition");
    periodic_boundary_condition(f_coll, f_equi, u, rho, rho_inlet, rho_outlet);
    // Advection
    //print("advection step");
    solver::advect(f_adve, f_coll);

    // Specular boundary conditions
    f_adve.index({Slice(), -1, 4}) = f_coll.index({Slice(), -1, 2});
    f_adve.index({Slice(), -1, 7}) = f_coll.index({Slice(), -1, 6});
    f_adve.index({Slice(), -1, 8}) = f_coll.index({Slice(), -1, 5});

    f_adve.index({Slice(), 0, 2}) = f_coll.index({Slice(), 0, 4});
    f_adve.index({Slice(), 0, 5}) = f_coll.index({Slice(), 0, 8});
    f_adve.index({Slice(), 0, 6}) = f_coll.index({Slice(), 0, 7});
  }

  // Save results
  print("saving results into files");
  torch::save(ux, "sbt-ux.pt");
  torch::save(uy, "sbt-uy.pt");
  torch::save(fs, "sbt-fs.pt");
  torch::save(rhos/3.0, "sbt-ps.pt");
  return 0;
}

