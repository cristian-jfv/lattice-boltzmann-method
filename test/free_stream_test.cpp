#include <ATen/TensorIndexing.h>
#include <iostream>
#include <ostream>
#include <torch/torch.h>
#include <toml++/toml.hpp>
#include "../src/params.hpp"
#include "../src/utils.hpp"
#include "../src/solver.hpp"

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
  toml::table tbl;
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

  Tensor f_equi = torch::zeros({lp.X, lp.Y, 9}, dev);
  Tensor f_coll = torch::zeros_like(f_equi, dev);
  Tensor f_adve = torch::zeros_like(f_equi, dev);
  Tensor u = torch::zeros({lp.X, lp.Y, 2}, dev);
  u.index({Ellipsis, 0}) = 0.1;
  Tensor rho = torch::ones({lp.X, lp.Y, 1}, dev);

  // Results
  Tensor ux = torch::zeros({lp.X, lp.Y, sp.total_snapshots});
  Tensor uy = torch::zeros_like(ux);
  Tensor rhos = torch::zeros_like(ux);

  // Parameters for the current study case
  const double p_grad = 8.0*lp.nu*lp.u/(lp.Y*lp.Y);
  const double rho_outlet = 1.0;
  const double rho_inlet = 1.6; //3.0*(lp.X-1)*p_grad + rho_outlet;
  utils::print("p_grad", p_grad);
  utils::print("rho_outlet", rho_outlet);
  utils::print("rho_inlet", rho_inlet);
  Tensor u_w = torch::zeros({lp.Y, 2}, dev);
  u_w.index({Slice(), 0}) = 0.1;
  Tensor abb_bc = torch::zeros({lp.Y, 1}, dev);

  if (argc < 3)
    if(!utils::continue_execution()) return 0;
  if (argc >= 3 && argv[2][0] == 'a')
    return 0;

  // Initialization
  solver::incomp_equilibrium(f_adve, u, rho);
  int i{0};
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
    solver::calc_incomp_u(u, f_adve);

    // Compute equilibrium
    solver::incomp_equilibrium(f_equi, u, rho);

    // Collision, BGK operator
    solver::collision(f_coll, f_adve, f_equi, lp.omega);

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
  utils::print("u_w", u_w);

  // Save results
  utils::print("\nSaving results");
  torch::save(ux, sp.file_prefix + "fst-ux.pt");
  torch::save(uy, sp.file_prefix + "fst-uy.pt");
  torch::save(rhos/3.0, sp.file_prefix + "fst-ps.pt");
  cout << "\a" << std::endl;
  return 0;
}
