#include <ATen/ops/ones_like.h>
#include <iostream>
#include <cmath>
#include <ostream>
#include <string>
#include <torch/torch.h>
#include <torch/types.h>

#include "../src/solver.hpp"
#include "../src/utils.hpp"
#include "../src/domain.hpp"

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
  solver::equilibrium(temp_equi, u.index(outlet).unsqueeze(0), rho_inlet*temp_rho);
  f_coll.index(virtual_inlet) = (temp_equi + f_coll.index(outlet) - f_equi.index(outlet)).squeeze(0).clone().detach();

  // outlet
  solver::equilibrium(temp_equi, u.index(inlet).unsqueeze(0), rho_outlet*temp_rho);
  f_coll.index(virtual_outlet) = (temp_equi + f_coll.index(inlet) - f_equi.index(inlet)).squeeze(0).clone().detach();
}

int main()
{
  // Flow parameters
  const int T = 500;
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

  // Domains
  domain A{H, W};
  A.m_0 = torch::ones_like(A.m_0);

  // Results
  Tensor A_fs = torch::zeros({H,W,9,T});
  Tensor A_ux = torch::zeros({H,W,T});
  Tensor A_uy = torch::zeros({H,W,T});
  Tensor A_rhos = torch::zeros({H,W,T});

  // Initialisation
  solver::equilibrium(A.adve_f, A.m_1, A.m_0);

  // Main loop
  print("main loop starts");
  for (int t=0; t < T; t++)
  {
    // cout << t << "\t\r" << std::flush;
    A_fs.index_put_({Ellipsis,t}, A.adve_f);
    A_ux.index_put_({Ellipsis,t}, A.m_1.index({Ellipsis,0}));
    A_uy.index_put_({Ellipsis,t}, A.m_1.index({Ellipsis,1}));
    A_rhos.index_put_({Ellipsis,t}, A.m_0.squeeze(2));

    // Calculate macroscopic variables
    solver::calc_rho(A.m_0, A.adve_f);
    solver::calc_u(A.m_1, A.adve_f, A.m_0);
    // Compute equilibrium
    solver::equilibrium(A.equi_f, A.m_1, A.m_0);
    // Collision, BGK operator
    solver::collision(A.coll_f, A.adve_f, A.equi_f, omega);
    // Inlet and outlet periodic boundary conditions
    periodic_boundary_condition(A.coll_f, A.equi_f, A.m_1, A.m_0, rho_inlet, rho_outlet);
    // Advection
    solver::advect(A.adve_f, A.coll_f);

    // No slip boundary conditions for the first and last column
    A.adve_f.index({Slice(), -1, 4}) = A.coll_f.index({Slice(), -1, 2}).clone().detach();
    A.adve_f.index({Slice(), -1, 7}) = A.coll_f.index({Slice(), -1, 5}).clone().detach();
    A.adve_f.index({Slice(), -1, 8}) = A.coll_f.index({Slice(), -1, 6}).clone().detach();

    A.adve_f.index({Slice(), 0, 2}) = A.coll_f.index({Slice(), 0, 4}).clone().detach();
    A.adve_f.index({Slice(), 0, 5}) = A.coll_f.index({Slice(), 0, 7}).clone().detach();
    A.adve_f.index({Slice(), 0, 6}) = A.coll_f.index({Slice(), 0, 8}).clone().detach();
  }

  // Save results
  std::string file_prefix = "A-domain-decomp-";
  print("saving results into files");
  torch::save(A_ux, file_prefix + "hpt-ux.pt");
  torch::save(A_uy, file_prefix + "hpt-uy.pt");
  torch::save(A_fs, file_prefix + "hpt-fs.pt");
  torch::save(A_rhos, file_prefix + "hpt-rho.pt");

  return 0;
}
