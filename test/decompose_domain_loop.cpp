#include <ATen/TensorIndexing.h>
#include <ATen/ops/ones_like.h>
#include <iostream>
#include <cmath>
#include <ostream>
#include <string>
#include <torch/csrc/autograd/generated/variable_factories.h>
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

const torch::Tensor E = torch::tensor(
  {4.0/ 9.0,
    1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0}, torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));
const torch::Tensor c = torch::tensor(
  {{0.0, 1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
   {0.0, 0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0}}, torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

int main()
{
  // Flow parameters
  const int T = 50000;
  const int snapshot_period = 50;
  print("T", T);
  const int L = 512;
  const int L2 = L/2;
  const int L4 = L/4;

  const double tau = std::sqrt(3.0/16.0) + 0.5;
  const double omega = 1.0/tau;
  cout << "omega=" << omega << endl;
  const double u_max = 0.1;
  const double nu = (2.0 * tau -1.0)/6.0;
  cout << "nu=" << nu << endl;
  const double Re = L4*u_max/nu;
  cout << "Re=" << Re << endl;

  torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(torch::kDouble));
  cout << "Default torch dtype: " << torch::get_default_dtype() << endl;
  if (!torch::cuda::is_available())
  {
    std::cerr << "CUDA is NOT available\n";
  }

  // Momentum source
  Tensor F = torch::tensor({{3E-3},{0.0}}, torch::kCUDA);
  const indices force_idx{Slice(L4+5,L4+55),Ellipsis};
  const double ics2 = 3.0;
  const double ics4 = 9.0;

  // Domains
  domain A{L,L4};
  domain B{L4,L2};
  domain C{L,L4};
  domain D{L4,L2};
  A.m_0 = torch::ones_like(A.m_0);
  B.m_0 = torch::ones_like(B.m_0);
  C.m_0 = torch::ones_like(C.m_0);
  D.m_0 = torch::ones_like(D.m_0);
  Tensor A_equi_populations = torch::zeros_like(A.coll_f, torch::kCUDA);

  // Results
  int H{A.R};
  int W{A.C};
  const int Ts = T/snapshot_period;
  Tensor A_ux = torch::zeros({H,W,Ts});
  Tensor A_uy = torch::zeros({H,W,Ts});
  Tensor A_rhos = torch::zeros({H,W,Ts});

  H = B.R;
  W = B.C;
  Tensor B_ux = torch::zeros({H,W,Ts});
  Tensor B_uy = torch::zeros({H,W,Ts});
  Tensor B_rhos = torch::zeros({H,W,Ts});

  H = C.R;
  W = C.C;
  Tensor C_ux = torch::zeros({H,W,Ts});
  Tensor C_uy = torch::zeros({H,W,Ts});
  Tensor C_rhos = torch::zeros({H,W,Ts});

  H = D.R;
  W = D.C;
  Tensor D_ux = torch::zeros({H,W,Ts});
  Tensor D_uy = torch::zeros({H,W,Ts});
  Tensor D_rhos = torch::zeros({H,W,Ts});

  // Initialisation
  solver::equilibrium(A.adve_f, A.m_1, A.m_0);
  solver::equilibrium(B.adve_f, B.m_1, B.m_0);
  solver::equilibrium(C.adve_f, C.m_1, C.m_0);
  solver::equilibrium(D.adve_f, D.m_1, D.m_0);

  // Main loop
  print("main loop starts");
  for (int t=0; t < T; t++)
  {
    A.m_1.index(force_idx) = A.m_1.index(force_idx) + F.t();
    // cout << t << "\t\r" << std::flush;
    if (t%snapshot_period == 0)
    {
      int ts = t/snapshot_period;
      A_ux.index_put_({Ellipsis,ts},   A.m_1.index({Ellipsis,0}));
      A_uy.index_put_({Ellipsis,ts},   A.m_1.index({Ellipsis,1}));
      A_rhos.index_put_({Ellipsis,ts}, A.m_0.squeeze(2));

      B_ux.index_put_({Ellipsis,ts},   B.m_1.index({Ellipsis,0}));
      B_uy.index_put_({Ellipsis,ts},   B.m_1.index({Ellipsis,1}));
      B_rhos.index_put_({Ellipsis,ts}, B.m_0.squeeze(2));

      C_ux.index_put_({Ellipsis,ts},   C.m_1.index({Ellipsis,0}));
      C_uy.index_put_({Ellipsis,ts},   C.m_1.index({Ellipsis,1}));
      C_rhos.index_put_({Ellipsis,ts}, C.m_0.squeeze(2));

      D_ux.index_put_({Ellipsis,ts},   D.m_1.index({Ellipsis,0}));
      D_uy.index_put_({Ellipsis,ts},   D.m_1.index({Ellipsis,1}));
      D_rhos.index_put_({Ellipsis,ts}, D.m_0.squeeze(2));
    }

    // Calculate macroscopic variables
    solver::calc_rho(A.m_0, A.adve_f);
    solver::calc_rho(B.m_0, B.adve_f);
    solver::calc_rho(C.m_0, C.adve_f);
    solver::calc_rho(D.m_0, D.adve_f);

    solver::calc_u(A.m_1, A.adve_f, A.m_0);
    solver::calc_u(B.m_1, B.adve_f, B.m_0);
    solver::calc_u(C.m_1, C.adve_f, C.m_0);
    solver::calc_u(D.m_1, D.adve_f, D.m_0);

    // Compute equilibrium
    solver::equilibrium(A.equi_f, A.m_1, A.m_0);
    solver::equilibrium(B.equi_f, B.m_1, B.m_0);
    solver::equilibrium(C.equi_f, C.m_1, C.m_0);
    solver::equilibrium(D.equi_f, D.m_1, D.m_0);
    // Force source terms
    Tensor S = ((1-0.5*omega)*((ics2 + ics4*A.m_1.index(force_idx).matmul(c))*F.t().matmul(c) - ics2*A.m_1.index(force_idx).matmul(F))*E).clone().detach();
    A_equi_populations.copy_(-omega*(A.adve_f - A.equi_f));


    // Collision, BGK operator
    //solver::collision(A.coll_f, A.adve_f, A.equi_f, omega);
    A.coll_f.copy_(A.adve_f + A_equi_populations);
    A.coll_f.index(force_idx) = (A.coll_f.index(force_idx)+S).detach().clone();
    solver::collision(B.coll_f, B.adve_f, B.equi_f, omega);
    solver::collision(C.coll_f, C.adve_f, C.equi_f, omega);
    solver::collision(D.coll_f, D.adve_f, D.equi_f, omega);

    // Advection
    solver::advect(A.adve_f, A.coll_f);
    solver::advect(B.adve_f, B.coll_f);
    solver::advect(C.adve_f, C.coll_f);
    solver::advect(D.adve_f, D.coll_f);

    // No slip boundary conditions

    // A
    // top
    A.adve_f.index_put_({0,Slice(),8}, A.coll_f.index({0,Slice(),6}));
    A.adve_f.index_put_({0,Slice(),1}, A.coll_f.index({0,Slice(),3}));
    A.adve_f.index_put_({0,Slice(),5}, A.coll_f.index({0,Slice(),7}));
    // bottom
    A.adve_f.index_put_({-1,Slice(),7}, A.coll_f.index({-1,Slice(),5}));
    A.adve_f.index_put_({-1,Slice(),3}, A.coll_f.index({-1,Slice(),1}));
    A.adve_f.index_put_({-1,Slice(),6}, A.coll_f.index({-1,Slice(),8}));
    // left
    A.adve_f.index({Slice(L4,-L4),0,2}) = A.coll_f.index({Slice(L4,-L4), 0, 4}).clone().detach();
    A.adve_f.index({Slice(L4,-L4),0,5}) = A.coll_f.index({Slice(L4,-L4), 0, 7}).clone().detach();
    A.adve_f.index({Slice(L4,-L4),0,6}) = A.coll_f.index({Slice(L4,-L4), 0, 8}).clone().detach();
    // right
    A.adve_f.index({Slice(1,-1), -1, 4}) = A.coll_f.index({Slice(1,-1), -1, 2}).clone().detach();
    A.adve_f.index({Slice(1,-1), -1, 7}) = A.coll_f.index({Slice(1,-1), -1, 5}).clone().detach();
    A.adve_f.index({Slice(1,-1), -1, 8}) = A.coll_f.index({Slice(1,-1), -1, 6}).clone().detach();

    // B
    // top
    B.adve_f.index_put_({0,Slice(),8}, B.coll_f.index({0,Slice(),6}));
    B.adve_f.index_put_({0,Slice(),1}, B.coll_f.index({0,Slice(),3}));
    B.adve_f.index_put_({0,Slice(),5}, B.coll_f.index({0,Slice(),7}));
    // bottom
    B.adve_f.index_put_({-1,Slice(),7}, B.coll_f.index({-1,Slice(),5}));
    B.adve_f.index_put_({-1,Slice(),3}, B.coll_f.index({-1,Slice(),1}));
    B.adve_f.index_put_({-1,Slice(),6}, B.coll_f.index({-1,Slice(),8}));

    // C
    // top
    C.adve_f.index_put_({0,Slice(),8}, C.coll_f.index({0,Slice(),6}));
    C.adve_f.index_put_({0,Slice(),1}, C.coll_f.index({0,Slice(),3}));
    C.adve_f.index_put_({0,Slice(),5}, C.coll_f.index({0,Slice(),7}));
    // bottom
    C.adve_f.index_put_({-1,Slice(),7}, C.coll_f.index({-1,Slice(),5}));
    C.adve_f.index_put_({-1,Slice(),3}, C.coll_f.index({-1,Slice(),1}));
    C.adve_f.index_put_({-1,Slice(),6}, C.coll_f.index({-1,Slice(),8}));
    // left
    C.adve_f.index({Slice(1,-1),0,2}) = C.coll_f.index({Slice(1,-1), 0, 4}).clone().detach();
    C.adve_f.index({Slice(1,-1),0,5}) = C.coll_f.index({Slice(1,-1), 0, 7}).clone().detach();
    C.adve_f.index({Slice(1,-1),0,6}) = C.coll_f.index({Slice(1,-1), 0, 8}).clone().detach();
    // right
    C.adve_f.index({Slice(L4,-L4), -1, 4}) = C.coll_f.index({Slice(L4,-L4), -1, 2}).clone().detach();
    C.adve_f.index({Slice(L4,-L4), -1, 7}) = C.coll_f.index({Slice(L4,-L4), -1, 5}).clone().detach();
    C.adve_f.index({Slice(L4,-L4), -1, 8}) = C.coll_f.index({Slice(L4,-L4), -1, 6}).clone().detach();

    // D
    // top
    D.adve_f.index_put_({0,Slice(),8}, D.coll_f.index({0,Slice(),6}));
    D.adve_f.index_put_({0,Slice(),1}, D.coll_f.index({0,Slice(),3}));
    D.adve_f.index_put_({0,Slice(),5}, D.coll_f.index({0,Slice(),7}));
    // bottom
    D.adve_f.index_put_({-1,Slice(),7}, D.coll_f.index({-1,Slice(),5}));
    D.adve_f.index_put_({-1,Slice(),3}, D.coll_f.index({-1,Slice(),1}));
    D.adve_f.index_put_({-1,Slice(),6}, D.coll_f.index({-1,Slice(),8}));

    // Bind the domains

    // A-B
    A.adve_f.index_put_({Slice(-L4,-1),0,6}, B.coll_f.index({Slice(1,None),-1,6}));
    A.adve_f.index_put_({Slice(-L4,None),0,2}, B.coll_f.index({Slice(),-1,2}));
    A.adve_f.index_put_({Slice(-L4+1,None),0,5}, B.coll_f.index({Slice(0,-1),-1,5}));
    B.adve_f.index_put_({Slice(1,None),-1,8}, A.coll_f.index({Slice(-L4,-1),0,8}));
    B.adve_f.index_put_({Slice(),-1,4}, A.coll_f.index({Slice(-L4,None),0,4}));
    B.adve_f.index_put_({Slice(0,-1),-1,7}, A.coll_f.index({Slice(-L4+1,None),0,7}));
    // B-C
    B.adve_f.index_put_({Slice(0,-1),0,6}, C.coll_f.index({Slice(-L4+1,None),-1,6}));
    B.adve_f.index_put_({Slice(),0,2}, C.coll_f.index({Slice(-L4,None),-1,2}));
    B.adve_f.index_put_({Slice(1,None),0,5}, C.coll_f.index({Slice(-L4,-1),-1,5}));
    C.adve_f.index_put_({Slice(-L4,-1),-1,7}, B.coll_f.index({Slice(1,None),0,7}));
    C.adve_f.index_put_({Slice(-L4,None),-1,4}, B.coll_f.index({Slice(),0,4}));
    C.adve_f.index_put_({Slice(-L4+1,None),-1,8}, B.coll_f.index({Slice(0,-1),0,8}));
    // C-D
    C.adve_f.index_put_({Slice(0,L4-1),-1,7}, D.coll_f.index({Slice(1,None),0,7}));
    C.adve_f.index_put_({Slice(0,L4),-1,4}, D.coll_f.index({Slice(),0,4}));
    C.adve_f.index_put_({Slice(1,L4),-1,8}, D.coll_f.index({Slice(0,-1),0,8}));
    D.adve_f.index_put_({Slice(0,-1),0,6}, C.coll_f.index({Slice(1,L4),-1,6}));
    D.adve_f.index_put_({Slice(),0,2}, C.coll_f.index({Slice(0,L4),-1,2}));
    D.adve_f.index_put_({Slice(1,None),0,5}, C.coll_f.index({Slice(0,L4-1),-1,5}));
    // D-A
    D.adve_f.index_put_({Slice(0,-1),-1,7}, A.coll_f.index({Slice(1,L4),0,7}));
    D.adve_f.index_put_({Slice(),-1,4}, A.coll_f.index({Slice(0,L4),0,4}));
    D.adve_f.index_put_({Slice(1,None),-1,8}, A.coll_f.index({Slice(0,L4-1),0,8}));
    A.adve_f.index_put_({Slice(0,L4-1),0,6}, D.coll_f.index({Slice(1,None),-1,6}));
    A.adve_f.index_put_({Slice(0,L4),0,2}, D.coll_f.index({Slice(),-1,2}));
    A.adve_f.index_put_({Slice(1,L4),0,5}, D.coll_f.index({Slice(0,-1),-1,5}));

  }

  // Save results
  std::string file_prefix = "A-domain-decomp-";
  print("saving results into files");
  torch::save(A_ux, file_prefix + "hpt-ux.pt");
  torch::save(A_uy, file_prefix + "hpt-uy.pt");
  torch::save(A_rhos, file_prefix + "hpt-rho.pt");

  file_prefix = "B-domain-decomp-";
  torch::save(B_ux, file_prefix + "hpt-ux.pt");
  torch::save(B_uy, file_prefix + "hpt-uy.pt");
  torch::save(B_rhos, file_prefix + "hpt-rho.pt");
  file_prefix = "C-domain-decomp-";
  torch::save(C_ux, file_prefix + "hpt-ux.pt");
  torch::save(C_uy, file_prefix + "hpt-uy.pt");
  torch::save(C_rhos, file_prefix + "hpt-rho.pt");
  file_prefix = "D-domain-decomp-";
  torch::save(D_ux, file_prefix + "hpt-ux.pt");
  torch::save(D_uy, file_prefix + "hpt-uy.pt");
  torch::save(D_rhos, file_prefix + "hpt-rho.pt");

  return 0;
}
