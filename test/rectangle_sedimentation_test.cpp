#include <ATen/TensorIndexing.h>
#include <iostream>
#include<ostream>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>
#include <toml++/toml.hpp>
#include "../src/params.hpp"
#include "../src/utils.hpp"
#include "../src/solver.hpp"

using torch::Tensor;
using torch::indexing::None;
using torch::indexing::Slice;
using torch::indexing::Ellipsis;
using std::cout;
using std::endl;
using std::cerr;

using utils::indices;
using solver::E;
using solver::c;

int main(int argc, char* argv[])
{
  // Read parameters
  toml::table tbl; // flow and simulation params
  try {
    tbl = toml::parse_file(argv[1]);
  } catch (const toml::parse_error& err) {
    cerr << "Parsing failed:\n" << err << "\n";
    return 1;
  }

  const params::flow fp{tbl};
  cout << fp << "\n";
  const params::lattice lp{tbl, fp};
  cout << lp << "\n";
  const params::simulation sp{tbl, lp};
  cout << sp << "\n";

  torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(torch::kDouble));
  if (!torch::cuda::is_available())
  {
    cerr << "CUDA is NOT available\n";
  }

  const torch::Device dev = torch::kCUDA;

  // Fluid dist. function
  Tensor f_equi = torch::zeros({lp.X, lp.Y, 9}, dev);
  Tensor f_coll = torch::zeros_like(f_equi, dev);
  Tensor f_adve = torch::zeros_like(f_equi, dev);
  Tensor u = torch::zeros({lp.X, lp.Y, 2}, dev);
  Tensor rho = torch::ones({lp.X, lp.Y, 1}, dev);

  // Sediment dist. function
  Tensor g_equi = torch::zeros({lp.X, lp.Y, 9}, dev);
  Tensor g_coll = torch::zeros_like(f_equi, dev);
  Tensor g_adve = torch::zeros_like(f_equi, dev);
  Tensor C = torch::zeros({lp.X,lp.Y,1}, dev);

  // Results
  Tensor ux = torch::zeros({lp.X, lp.Y, sp.total_snapshots});
  Tensor uy = torch::zeros_like(ux);
  Tensor rhos = torch::zeros_like(ux);
  Tensor Cs = torch::zeros_like(ux);

  // Parameters for the current study case
  utils::print("\nParameters for the current study case");
  // const double ics2 = 3.0;

  // Wall coordinates
  const int R23 = -151; //(int)(lp.X*2/3);
  const int C28 = 200; //(int)(lp.Y*2/8);
  const int C38 = 250; //(int)(lp.Y*5/16);
  cout << R23 << "\n" << C28 << "\n" << C38 << std::endl;

  // Inlet wall velocity (fixed)
  Tensor fixed_u_w = torch::zeros({lp.X, 2}, dev);
  fixed_u_w.index({Slice(), 1}) = lp.u;
  // Outlet wall velocity
  Tensor u_w = torch::zeros({lp.X, 2}, dev);
  u.index({Ellipsis, 1}) = lp.u;
  Tensor bb_bc = torch::zeros({lp.X, 9}, dev);
  Tensor abb_bc = torch::zeros({lp.X, 9}, dev);
  Tensor g_abb_bc = torch::zeros({lp.X, 9}, dev);

  // Sediment dist. function initialization
  const double w_s = 3e-3;
  const double scalar_C_w = 1e-3; //1.0/lp.X;
  utils::print("C_w", scalar_C_w);
  Tensor C_w = torch::zeros({lp.X}, dev);
  C_w.index({Slice(-50,None)}) = scalar_C_w;
  C.index({Slice(),0,0}) = C_w;
  solver::equilibrium(g_adve, u, C);

  // if(!utils::continue_execution()) return 1;

  // Initialization
  solver::incomp_equilibrium(f_adve, u, rho);

  // Calculate macroscopic variables
  solver::calc_rho(rho, f_adve);
  solver::calc_u(u, f_adve, rho);
  solver::equilibrium(g_equi, u, C);

  int i{0};
  cout << torch::tensor({0});
  cout << "main loop\n";
  for (int t=0; t<sp.total_steps; t++)
  {
    if(sp.snapshot(t))
    {
      cout << t << "; t=" << t*lp.dt << " s\t\t\r" << std::flush;
      // Save snapshot
      ux.index({Ellipsis,i}) = u.index({Ellipsis, 0}).clone().detach();
      uy.index({Ellipsis,i}) = u.index({Ellipsis, 1}).clone().detach();
      rhos.index({Ellipsis,i}) = rho.squeeze(2).clone().detach();
      Cs.index({Ellipsis,i}) = C.squeeze(-1).detach().clone();
      ++i;
    }

    // Compute equilibrium
    solver::equilibrium(f_equi, u, rho);
    solver::equilibrium(g_equi, u+w_s, C);

    // Collision, BGK operator
    // f_coll.copy_(
    //   f_adve
    // );
    solver::collision(f_coll, f_adve, f_equi, lp.omega);
    solver::collision(g_coll, g_adve, g_equi, lp.omega/1.0); // Sc = 1.0

    // Zero gradient boundary condition:
    // Copy the post-collision populations to the outlet
    // plane before propagation
    // Top
    g_coll.index_put_({0, Ellipsis}, g_coll.index({1, Ellipsis}));
    // Outlet
    g_coll.index_put_({Slice(1,-1),-1,Slice()},
                      g_coll.index({Slice(1,-1),-2,Slice()}));

    // Advection
    solver::advect(f_adve, f_coll);
    solver::advect(g_adve, g_coll);

    // Anti-bounce-back boundary conditions
    // Inlet
    //bb_bc.copy_( -2.0*ics2*u_w.matmul(c)*E );
    abb_bc.copy_( ((2.0 + 9.0*torch::pow(fixed_u_w.matmul(c),2.0) - 3.0*fixed_u_w.mul(fixed_u_w).sum(1).unsqueeze(1))*E).squeeze(0) );
    // f_adve.index_put_({Slice(1,-1), 0, 5}, -f_coll.index({Slice(1,-1), 0, 7}) + abb_bc.index({Slice(1,-1), 7}));
    // f_adve.index_put_({Slice(1,-1), 0, 2}, -f_coll.index({Slice(1,-1), 0, 4}) + abb_bc.index({Slice(1,-1), 4}));
    // f_adve.index_put_({Slice(1,-1), 0, 6}, -f_coll.index({Slice(1,-1), 0, 8}) + abb_bc.index({Slice(1,-1), 8}));
    f_adve.index_put_({Slice(1,-1), 0, 3}, -f_coll.index({Slice(1,-1), 0, 1}) + abb_bc.index({Slice(1,-1), 1}));
    f_adve.index_put_({Slice(1,-1), 0, 4}, -f_coll.index({Slice(1,-1), 0, 2}) + abb_bc.index({Slice(1,-1), 2}));
    f_adve.index_put_({Slice(1,-1), 0, 1}, -f_coll.index({Slice(1,-1), 0, 3}) + abb_bc.index({Slice(1,-1), 3}));
    f_adve.index_put_({Slice(1,-1), 0, 2}, -f_coll.index({Slice(1,-1), 0, 4}) + abb_bc.index({Slice(1,-1), 4}));
    f_adve.index_put_({Slice(1,-1), 0, 7}, -f_coll.index({Slice(1,-1), 0, 5}) + abb_bc.index({Slice(1,-1), 5}));
    f_adve.index_put_({Slice(1,-1), 0, 8}, -f_coll.index({Slice(1,-1), 0, 6}) + abb_bc.index({Slice(1,-1), 6}));
    f_adve.index_put_({Slice(1,-1), 0, 5}, -f_coll.index({Slice(1,-1), 0, 7}) + abb_bc.index({Slice(1,-1), 7}));
    f_adve.index_put_({Slice(1,-1), 0, 6}, -f_coll.index({Slice(1,-1), 0, 8}) + abb_bc.index({Slice(1,-1), 8}));
    // Outlet
    u_w = (1.5*u.index({Slice(), -1}) - 0.5*u.index({Slice(), -2})).clone().detach();
    abb_bc.copy_( ((2.0 + 9.0*torch::pow(u_w.matmul(c),2.0) - 3.0*u_w.mul(u_w).sum(1).unsqueeze(1))*E).squeeze(0) );
    f_adve.index_put_({Slice(), -1, 3}, -f_coll.index({Slice(), -1, 1}) + abb_bc.index({Slice(), 1}));
    f_adve.index_put_({Slice(), -1, 4}, -f_coll.index({Slice(), -1, 2}) + abb_bc.index({Slice(), 2}));
    f_adve.index_put_({Slice(), -1, 1}, -f_coll.index({Slice(), -1, 3}) + abb_bc.index({Slice(), 3}));
    f_adve.index_put_({Slice(), -1, 2}, -f_coll.index({Slice(), -1, 4}) + abb_bc.index({Slice(), 4}));
    f_adve.index_put_({Slice(), -1, 7}, -f_coll.index({Slice(), -1, 5}) + abb_bc.index({Slice(), 5}));
    f_adve.index_put_({Slice(), -1, 8}, -f_coll.index({Slice(), -1, 6}) + abb_bc.index({Slice(), 6}));
    f_adve.index_put_({Slice(), -1, 5}, -f_coll.index({Slice(), -1, 7}) + abb_bc.index({Slice(), 7}));
    f_adve.index_put_({Slice(), -1, 6}, -f_coll.index({Slice(), -1, 8}) + abb_bc.index({Slice(), 8}));

    // Specular top
    f_adve.index_put_({0, Slice(), 8}, f_coll.index({0, Slice(), 7}));
    f_adve.index_put_({0, Slice(), 1}, f_coll.index({0, Slice(), 3}));
    f_adve.index_put_({0, Slice(), 5}, f_coll.index({0, Slice(), 6}));

    // Bottom: no slip
    f_adve.index_put_({-1, Slice(), 7}, f_coll.index({-1, Slice(), 5}));
    f_adve.index_put_({-1, Slice(), 3}, f_coll.index({-1, Slice(), 1}));
    f_adve.index_put_({-1, Slice(), 6}, f_coll.index({-1, Slice(), 8}));

    // Rectangle boundaries: no slip
    // First wall
    f_adve.index_put_({Slice(R23+1,-1),C28,8}, f_coll.index({Slice(R23+1,-1),C28,6}));
    f_adve.index_put_({Slice(R23+1,-1),C28,4}, f_coll.index({Slice(R23+1,-1),C28,2}));
    f_adve.index_put_({Slice(R23+1,-1),C28,7}, f_coll.index({Slice(R23+1,-1),C28,5}));
    // Ceiling
    f_adve.index_put_({R23,Slice(C28,C38+1),6}, f_coll.index({R23,Slice(C28,C38+1),8}));
    f_adve.index_put_({R23,Slice(C28,C38+1),3}, f_coll.index({R23,Slice(C28,C38+1),1}));
    f_adve.index_put_({R23,Slice(C28,C38+1),7}, f_coll.index({R23,Slice(C28,C38+1),5}));
    // Second wall
    f_adve.index_put_({Slice(R23+1,-1),C38,5}, f_coll.index({Slice(R23+1,-1),C38,7}));
    f_adve.index_put_({Slice(R23+1,-1),C38,2}, f_coll.index({Slice(R23+1,-1),C38,4}));
    f_adve.index_put_({Slice(R23+1,-1),C38,6}, f_coll.index({Slice(R23+1,-1),C38,8}));

    // Calculate macroscopic variables
    solver::calc_rho(rho, f_adve);
    solver::calc_u(u, f_adve, rho);
    solver::equilibrium(g_equi, u+w_s, C);

    // ADE Inlet
    g_abb_bc.copy_(
      ( 1.0
      + 3.0*matmul(u.index({Slice(),0,Slice()})+w_s,c)
      + 4.5*matmul(u.index({Slice(),0,Slice()})+w_s,c).pow(2)
      - 1.5*((u.index({Slice(),0,Slice()})+w_s)*(u.index({Slice(),0,Slice()})+w_s)).sum(-1).unsqueeze(-1)
      )*E*C_w.unsqueeze(-1)
    );
    g_adve.index_put_({Slice(1,-1), 0, 3}, -g_coll.index({Slice(1,-1), 0, 1}) + 2.0*g_abb_bc.index({Slice(1,-1), 1}));
    g_adve.index_put_({Slice(1,-1), 0, 4}, -g_coll.index({Slice(1,-1), 0, 2}) + 2.0*g_abb_bc.index({Slice(1,-1), 2}));
    g_adve.index_put_({Slice(1,-1), 0, 1}, -g_coll.index({Slice(1,-1), 0, 3}) + 2.0*g_abb_bc.index({Slice(1,-1), 3}));
    g_adve.index_put_({Slice(1,-1), 0, 2}, -g_coll.index({Slice(1,-1), 0, 4}) + 2.0*g_abb_bc.index({Slice(1,-1), 4}));
    g_adve.index_put_({Slice(1,-1), 0, 7}, -g_coll.index({Slice(1,-1), 0, 5}) + 2.0*g_abb_bc.index({Slice(1,-1), 5}));
    g_adve.index_put_({Slice(1,-1), 0, 8}, -g_coll.index({Slice(1,-1), 0, 6}) + 2.0*g_abb_bc.index({Slice(1,-1), 6}));
    g_adve.index_put_({Slice(1,-1), 0, 5}, -g_coll.index({Slice(1,-1), 0, 7}) + 2.0*g_abb_bc.index({Slice(1,-1), 7}));
    g_adve.index_put_({Slice(1,-1), 0, 6}, -g_coll.index({Slice(1,-1), 0, 8}) + 2.0*g_abb_bc.index({Slice(1,-1), 8}));

    // Rectangle boundaries: no slip
    // First wall
    g_adve.index_put_({Slice(R23+1,None),C28,8}, -g_coll.index({Slice(R23+1,None),C28,6}));
    g_adve.index_put_({Slice(R23+1,None),C28,4}, -g_coll.index({Slice(R23+1,None),C28,2}));
    g_adve.index_put_({Slice(R23+1,None),C28,7}, -g_coll.index({Slice(R23+1,None),C28,5}));
    // Ceiling
    g_adve.index_put_({R23,Slice(C28,C38+1),6}, -g_coll.index({R23,Slice(C28,C38+1),8}));
    g_adve.index_put_({R23,Slice(C28,C38+1),3}, -g_coll.index({R23,Slice(C28,C38+1),1}));
    g_adve.index_put_({R23,Slice(C28,C38+1),7}, -g_coll.index({R23,Slice(C28,C38+1),5}));
    // Second wall
    g_adve.index_put_({Slice(R23+1,-1),C38,5}, -g_coll.index({Slice(R23+1,-1),C38,7}));
    g_adve.index_put_({Slice(R23+1,-1),C38,2}, -g_coll.index({Slice(R23+1,-1),C38,4}));
    g_adve.index_put_({Slice(R23+1,-1),C38,6}, -g_coll.index({Slice(R23+1,-1),C38,8}));
    // Bottom
    g_adve.index_put_({-1,Slice(),6}, g_coll.index({-1,Slice(),8}));
    g_adve.index_put_({-1,Slice(),3}, g_coll.index({-1,Slice(),1}));
    g_adve.index_put_({-1,Slice(),7}, g_coll.index({-1,Slice(),5}));
    solver::calc_rho(C, g_adve);
  }

  // Save results
  utils::print("\nSaving results");
  torch::save(ux, sp.file_prefix + "-ux.pt");
  torch::save(uy, sp.file_prefix + "-uy.pt");
  torch::save(rhos/3.0, sp.file_prefix + "-ps.pt");
  torch::save(Cs, sp.file_prefix + "-cs.pt");

  return 0;
}
