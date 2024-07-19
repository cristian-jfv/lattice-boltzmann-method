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
using torch::indexing::None;
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

void periodic_boundary_condition
(
  domain& A,
  domain& B,
  const double rho_inlet,
  const double rho_outlet
)
{
  torch::Tensor temp_equi = torch::zeros({1, A.C, 9}, torch::kCUDA);
  torch::Tensor temp_rho = torch::ones({1, A.C, 1}, torch::kCUDA);

  // inlet
  solver::equilibrium(temp_equi, B.m_1.index(outlet).unsqueeze(0), rho_inlet*temp_rho);
  A.coll_f.index_put_(virtual_inlet,
    (temp_equi + B.coll_f.index(outlet) - B.equi_f.index(outlet)).squeeze(0)
  );

  // outlet
  solver::equilibrium(temp_equi, A.m_1.index(inlet).unsqueeze(0), rho_outlet*temp_rho);
  B.coll_f.index_put_(virtual_outlet,
    (temp_equi + A.coll_f.index(inlet) - A.equi_f.index(inlet)).squeeze(0)
  );

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
  domain B{H, W};
  B.m_0 = torch::ones_like(B.m_0);

  // Results
  Tensor A_fs = torch::zeros({H,W,9,T});
  Tensor A_ux = torch::zeros({H,W,T});
  Tensor A_uy = torch::zeros({H,W,T});
  Tensor A_rhos = torch::zeros({H,W,T});

  Tensor B_fs = torch::zeros({H,W,9,T});
  Tensor B_ux = torch::zeros({H,W,T});
  Tensor B_uy = torch::zeros({H,W,T});
  Tensor B_rhos = torch::zeros({H,W,T});

  // Initialisation
  solver::equilibrium(A.adve_f, A.m_1, A.m_0);
  solver::equilibrium(B.adve_f, B.m_1, B.m_0);

  // Main loop
  print("main loop starts");
  for (int t=0; t < T; t++)
  {
    // cout << t << "\t\r" << std::flush;
    A_fs.index_put_({Ellipsis,t},   A.adve_f);
    A_ux.index_put_({Ellipsis,t},   A.m_1.index({Ellipsis,0}));
    A_uy.index_put_({Ellipsis,t},   A.m_1.index({Ellipsis,1}));
    A_rhos.index_put_({Ellipsis,t}, A.m_0.squeeze(2));

    B_fs.index_put_({Ellipsis,t},   B.adve_f);
    B_ux.index_put_({Ellipsis,t},   B.m_1.index({Ellipsis,0}));
    B_uy.index_put_({Ellipsis,t},   B.m_1.index({Ellipsis,1}));
    B_rhos.index_put_({Ellipsis,t}, B.m_0.squeeze(2));

    // Calculate macroscopic variables
    solver::calc_rho(A.m_0, A.adve_f);
    solver::calc_rho(B.m_0, B.adve_f);
    solver::calc_u(A.m_1, A.adve_f, A.m_0);
    solver::calc_u(B.m_1, B.adve_f, B.m_0);

    // Compute equilibrium
    solver::equilibrium(A.equi_f, A.m_1, A.m_0);
    solver::equilibrium(B.equi_f, B.m_1, B.m_0);

    // Collision, BGK operator
    solver::collision(A.coll_f, A.adve_f, A.equi_f, omega);
    solver::collision(B.coll_f, B.adve_f, B.equi_f, omega);

    // Inlet and outlet periodic boundary conditions
    // periodic_boundary_condition(A.coll_f, A.equi_f, A.m_1, A.m_0, rho_inlet, rho_outlet);
    periodic_boundary_condition(A, B, rho_inlet, rho_outlet);

    // Advection
    solver::advect(A.adve_f, A.coll_f);
    solver::advect(B.adve_f, B.coll_f);

    // No slip boundary conditions for the first and last column
    A.adve_f.index({Slice(), -1, 4}) = A.coll_f.index({Slice(), -1, 2}).clone().detach();
    A.adve_f.index({Slice(), -1, 7}) = A.coll_f.index({Slice(), -1, 5}).clone().detach();
    A.adve_f.index({Slice(), -1, 8}) = A.coll_f.index({Slice(), -1, 6}).clone().detach();

    A.adve_f.index({Slice(), 0, 2}) = A.coll_f.index({Slice(), 0, 4}).clone().detach();
    A.adve_f.index({Slice(), 0, 5}) = A.coll_f.index({Slice(), 0, 7}).clone().detach();
    A.adve_f.index({Slice(), 0, 6}) = A.coll_f.index({Slice(), 0, 8}).clone().detach();

    // No slip boundary conditions for the B domain
    B.adve_f.index({Slice(), -1, 4}) = B.coll_f.index({Slice(), -1, 2}).clone().detach();
    B.adve_f.index({Slice(), -1, 7}) = B.coll_f.index({Slice(), -1, 5}).clone().detach();
    B.adve_f.index({Slice(), -1, 8}) = B.coll_f.index({Slice(), -1, 6}).clone().detach();

    B.adve_f.index({Slice(), 0, 2}) = B.coll_f.index({Slice(), 0, 4}).clone().detach();
    B.adve_f.index({Slice(), 0, 5}) = B.coll_f.index({Slice(), 0, 7}).clone().detach();
    B.adve_f.index({Slice(), 0, 6}) = B.coll_f.index({Slice(), 0, 8}).clone().detach();

    // Bind the domains
    A.adve_f.index_put_({-1,Slice(),3}, B.coll_f.index({0,Slice(),3}).detach());
    A.adve_f.index_put_({-1,Slice(1,None),6}, B.coll_f.index({0,Slice(0,-1),6}).detach());
    A.adve_f.index_put_({-1,Slice(0,-1),7}, B.coll_f.index({0,Slice(1,None),7}).detach());

    B.adve_f.index_put_({0,Slice(),1}, A.coll_f.index({-1,Slice(),1}).detach() );
    B.adve_f.index_put_({0,Slice(1,None),5}, A.coll_f.index({-1,Slice(0,-1),5}).detach() );
    B.adve_f.index_put_({0,Slice(0,-1),8}, A.coll_f.index({-1,Slice(1,None),8}).detach() );
  }

  // Save results
  std::string file_prefix = "A-domain-decomp-";
  print("saving results into files");
  torch::save(A_ux, file_prefix + "hpt-ux.pt");
  torch::save(A_uy, file_prefix + "hpt-uy.pt");
  torch::save(A_fs, file_prefix + "hpt-fs.pt");
  torch::save(A_rhos, file_prefix + "hpt-rho.pt");

  file_prefix = "B-domain-decomp-";
  torch::save(B_ux, file_prefix + "hpt-ux.pt");
  torch::save(B_uy, file_prefix + "hpt-uy.pt");
  torch::save(B_fs, file_prefix + "hpt-fs.pt");
  torch::save(B_rhos, file_prefix + "hpt-rho.pt");
  return 0;
}
