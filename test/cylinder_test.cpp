#include <iostream>
#include <torch/torch.h>
#include <toml++/toml.hpp>
#include "../src/params.hpp"
#include "../src/utils.hpp"

using torch::Tensor;
using std::cout;
using std::cerr;

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

  if(!utils::continue_execution()) return 0;

  Tensor f = torch::zeros({lp.X, lp.Y, 9});

  return 0;
}
