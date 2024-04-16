#include <cassert>
#include <iostream>
#include <torch/torch.h>

#include "../src/domain.hpp"

using std::cout;
using std::endl;

using torch::Tensor;

int test_boundary_view();

int main()
{
  torch::manual_seed(33156); // reproducible results
  int r = 0;
  r = test_boundary_view();
  assert(r==0);

  return 0;
}

int test_boundary_view()
{
  Tensor f = torch::rand({3, 4, 9});
  Tensor left = domain::left_boundary(f);
  Tensor right = domain::right_boundary(f);
  Tensor top = domain::top_boundary(f);
  Tensor bottom = domain::bottom_boundary(f);

  cout << "f=\n" << f << endl;
  cout << "left=\n" << left << endl;
  cout << "right=\n" << right << endl;
  cout << "top=\n" << top << endl;
  cout << "bottom=\n" << bottom << endl;

  return 0;
}
