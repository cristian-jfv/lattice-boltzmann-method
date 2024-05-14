#include <ATen/TensorIndexing.h>
#include <iostream>
#include<ostream>
#include <torch/torch.h>
#include <toml++/toml.hpp>
#include "../src/params.hpp"
#include "../src/utils.hpp"
#include "../src/solver.hpp"
#include "../src/ibm.hpp"

using torch::Tensor;
using torch::indexing::Slice;
using torch::indexing::Ellipsis;
using std::cout;
using std::cerr;

using utils::indices;
using solver::E;
using solver::c;

int main(int argc, char* argv[])
{
  // Read parameters
  toml::table tbl; // flow and simulation params
  toml::table tbl_boundary; // immersed boundary data
  try {
    tbl = toml::parse_file(argv[1]);
    tbl_boundary = toml::parse_file(argv[2]);
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

  Tensor f_equi = torch::zeros({lp.X, lp.Y, 9}, dev);
  Tensor f_coll = torch::zeros_like(f_equi, dev);
  Tensor f_adve = torch::zeros_like(f_equi, dev);
  Tensor u = torch::zeros({lp.X, lp.Y, 2}, dev);
  Tensor rho = torch::ones({lp.X, lp.Y, 1}, dev);

  // Results
  Tensor ux = torch::zeros({lp.X, lp.Y, sp.total_snapshots});
  Tensor uy = torch::zeros_like(ux);
  Tensor rhos = torch::zeros_like(ux);

  // Immersed boundary configuration
  ibm ib{tbl_boundary, "cylinder-a", dev};
  Tensor F, S;
  const double ics2 = 1.0/3.0;
  const double ics4 = 1.0/9.0;
  Tensor equi_populations = torch::zeros_like(f_equi);

  // Parameters for the current study case
  const double p_grad = 8.0*lp.nu*lp.u/(lp.Y*lp.Y);
  const double rho_outlet = 1.0;
  const double rho_inlet = 1.6; //3.0*(lp.X-1)*p_grad + rho_outlet; //1.6;
  utils::print("\nParameters for the current study case");
  utils::print("p_grad", p_grad);
  utils::print("rho_outlet", rho_outlet);
  utils::print("rho_inlet", rho_inlet);
  Tensor u_w = torch::zeros({lp.Y, 2}, dev);
  u_w.index({Slice(), 0}) = lp.u;
  u.index({Ellipsis, 0}) = lp.u;
  Tensor abb_bc = torch::zeros({lp.Y, 1}, dev);
  cout << torch::ones({1,5});

  if (argc < 4)
    if(!utils::continue_execution()) return 0;
  if (argc >= 4 && argv[3][0] == 'a')
    return 0;

  // Initialization
  solver::incomp_equilibrium(f_adve, u, rho);
  int i{0};
  cout << "\n";
  for (int t=0; t<sp.total_steps; t++)
  {
    if(sp.snapshot(t))
    {
      cout << t << "; t=" << t*lp.dt << " s\t\r" << std::flush;
      // Save snapshot
      ux.index({Ellipsis,i}) = u.index({Ellipsis, 0}).clone().detach();
      uy.index({Ellipsis,i}) = u.index({Ellipsis, 1}).clone().detach();
      rhos.index({Ellipsis,i}) = rho.squeeze(2).clone().detach();
      ++i;
    }

    // Calculate macroscopic variables
    solver::calc_rho(rho, f_adve);
    solver::calc_u(u, f_adve, rho);

    // Compute equilibrium
    solver::equilibrium(f_equi, u, rho);
    equi_populations.copy_(-lp.omega*( f_adve - f_equi ));

    F = ib.eulerian_force_density(u, rho);

    // Force source terms
    auto u_roi = u.index({ib.rows, ib.cols, Slice()});
    S = ((1-0.5*lp.omega)*((ics2 + ics4*u_roi.matmul(c))*F.matmul(c)
      - ics2*(u_roi*F).sum(2).unsqueeze(2))
      *E).clone().detach();

    // Collision, BGK operator
    //print("collision step");
    f_coll.copy_(
      f_adve + equi_populations
    );

    f_coll.index({ib.rows, ib.cols, Slice()}) += S;

    // Advection
    solver::advect(f_adve, f_coll);

    // Anti-bounce-back boundary conditions
    // Inlet
    //u_w = (1.5*u.index({0, Ellipsis}) - 0.5*u.index({1, Ellipsis})).clone().detach();
    abb_bc = ((2.0 + 9.0*torch::pow(u_w.matmul(c),2.0) - 3.0*u_w.mul(u_w).sum(1).unsqueeze(1))*E).squeeze(0).clone().detach();
    f_adve.index({0, Slice(), 3}) = (-f_coll.index({0, Slice(), 1}) + abb_bc.index({Slice(), 1})).clone().detach();
    f_adve.index({0, Slice(), 4}) = (-f_coll.index({0, Slice(), 2}) + abb_bc.index({Slice(), 2})).clone().detach();
    f_adve.index({0, Slice(), 1}) = (-f_coll.index({0, Slice(), 3}) + abb_bc.index({Slice(), 3})).clone().detach();
    f_adve.index({0, Slice(), 2}) = (-f_coll.index({0, Slice(), 4}) + abb_bc.index({Slice(), 4})).clone().detach();
    f_adve.index({0, Slice(), 7}) = (-f_coll.index({0, Slice(), 5}) + abb_bc.index({Slice(), 5})).clone().detach();
    f_adve.index({0, Slice(), 8}) = (-f_coll.index({0, Slice(), 6}) + abb_bc.index({Slice(), 6})).clone().detach();
    f_adve.index({0, Slice(), 5}) = (-f_coll.index({0, Slice(), 7}) + abb_bc.index({Slice(), 7})).clone().detach();
    f_adve.index({0, Slice(), 6}) = (-f_coll.index({0, Slice(), 8}) + abb_bc.index({Slice(), 8})).clone().detach();
    // Outlet
    //u_w = (1.5*u.index({-1, Ellipsis}) - 0.5*u.index({-2, Ellipsis})).clone().detach();
    abb_bc = ((2.0 + 9.0*torch::pow(u_w.matmul(c),2.0) - 3.0*u_w.mul(u_w).sum(1).unsqueeze(1))*E).squeeze(0).clone().detach();
    f_adve.index({-1, Slice(), 3}) = (-f_coll.index({-1, Slice(), 1}) + abb_bc.index({Slice(), 1})).clone().detach();
    f_adve.index({-1, Slice(), 4}) = (-f_coll.index({-1, Slice(), 2}) + abb_bc.index({Slice(), 2})).clone().detach();
    f_adve.index({-1, Slice(), 1}) = (-f_coll.index({-1, Slice(), 3}) + abb_bc.index({Slice(), 3})).clone().detach();
    f_adve.index({-1, Slice(), 2}) = (-f_coll.index({-1, Slice(), 4}) + abb_bc.index({Slice(), 4})).clone().detach();
    f_adve.index({-1, Slice(), 7}) = (-f_coll.index({-1, Slice(), 5}) + abb_bc.index({Slice(), 5})).clone().detach();
    f_adve.index({-1, Slice(), 8}) = (-f_coll.index({-1, Slice(), 6}) + abb_bc.index({Slice(), 6})).clone().detach();
    f_adve.index({-1, Slice(), 5}) = (-f_coll.index({-1, Slice(), 7}) + abb_bc.index({Slice(), 7})).clone().detach();
    f_adve.index({-1, Slice(), 6}) = (-f_coll.index({-1, Slice(), 8}) + abb_bc.index({Slice(), 8})).clone().detach();

    // Specular boundary conditions
    f_adve.index({Slice(), -1, 4}) = f_coll.index({Slice(), -1, 2}).clone().detach();
    f_adve.index({Slice(), -1, 7}) = f_coll.index({Slice(), -1, 6}).clone().detach();
    f_adve.index({Slice(), -1, 8}) = f_coll.index({Slice(), -1, 5}).clone().detach();

    f_adve.index({Slice(), 0, 2}) = f_coll.index({Slice(), 0, 4}).clone().detach();
    f_adve.index({Slice(), 0, 5}) = f_coll.index({Slice(), 0, 8}).clone().detach();
    f_adve.index({Slice(), 0, 6}) = f_coll.index({Slice(), 0, 7}).clone().detach();
  }

  // Save results
  utils::print("\nSaving results");
  torch::save(ux, sp.file_prefix + "ct-ux.pt");
  torch::save(uy, sp.file_prefix + "ct-uy.pt");
  torch::save(rhos/3.0, sp.file_prefix + "ct-ps.pt");

  return 0;
}
