#include <ATen/ops/zeros_like.h>
#include <c10/core/DeviceType.h>
#include <iostream>
#include <ostream>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/cuda.h>
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
);

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
  // Parameters for the current study case
  // Hagen-Poiseuille pressure drop
  const double p_grad = 8.0*lp.nu*lp.u/(lp.Y*lp.Y);
  const double rho_outlet = 1.0;
  const double rho_inlet = 3.0*(lp.X - 1)*p_grad + rho_outlet;
  utils::print("p_grad", p_grad);
  utils::print("rho_outlet", rho_outlet);
  utils::print("rho_inlet", rho_inlet);

  if (argc < 3)
    if(!utils::continue_execution()) return 0;
  if (argc >= 3 && argv[2][0] == 'a')
    return 0;

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
  Tensor rho = torch::zeros({lp.X, lp.Y, 1}, dev);

  // Results
  Tensor ux = torch::zeros({lp.X, lp.Y, sp.total_snapshots});
  Tensor uy = torch::zeros_like(ux);
  Tensor rhos = torch::zeros_like(ux);

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

    // Inlet and outlet periodic boundary conditions
    periodic_boundary_condition(f_coll, f_equi, u, rho, rho_inlet, rho_outlet);

    // Advection
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
  utils::print("\nSaving results");
  torch::save(ux, sp.file_prefix + "ct-ux.pt");
  torch::save(uy, sp.file_prefix + "ct-uy.pt");
  torch::save(rhos/3.0, sp.file_prefix + "ct-ps.pt");

  return 0;
}

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
