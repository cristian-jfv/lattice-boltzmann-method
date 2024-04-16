#include "domain.hpp"
#include <iostream>

namespace domain
{
using torch::indexing::Slice;
using std::cout;
using std::endl;

torch::Tensor left_boundary(torch::Tensor& domain)
{ return domain.index({Slice(), 0, Slice()}); }

torch::Tensor right_boundary(torch::Tensor& domain)
{ return domain.index({Slice(), -1, Slice()}); }

torch::Tensor top_boundary(torch::Tensor& domain)
{ return domain.index({0, Slice(), Slice()}); }

torch::Tensor bottom_boundary(torch::Tensor& domain)
{ return domain.index({-1, Slice(), Slice()}); }

void no_slip(torch::Tensor& boundary, interface itf)
{
  // Work columnwise
  // Assume the wall velocity is zero
  // TODO: Implement moving wall no slip condition

  if (itf==interface::wall_to_fluid)
  {
    // f3 = f5
    boundary.index({Slice(1,-1), 3-1}) =
      boundary.index({Slice(1,-1), 5-1}).clone().detach();
    // f6 = f8
    boundary.index({Slice(1,-1), 6-1}) =
      boundary.index({Slice(1,-1), 8-1}).clone().detach();
    // f7 = f9
    boundary.index({Slice(1,-1), 7-1}) =
      boundary.index({Slice(1,-1), 9-1}).clone().detach();
  }
  else if (itf==interface::fluid_to_wall)
  {
    // f5 = f3
    boundary.index({Slice(1,-1), 5-1}) =
      boundary.index({Slice(1,-1), 3-1}).clone().detach();
    // f8 = f6
    boundary.index({Slice(1,-1), 8-1}) =
      boundary.index({Slice(1,-1), 6-1}).clone().detach();
    // f9 = f7
    boundary.index({Slice(1,-1), 9-1}) =
      boundary.index({Slice(1,-1), 7-1}).clone().detach();
  }
}

torch::Tensor C(const torch::Tensor& inlet, const torch::Tensor& outlet, double dp)
{
  // Assume inlet and outlet have rank 2
  return dp - (1.0/3.0)*(
    inlet.index({Slice(), 1-1}) - outlet.index({Slice(), 1-1})
    + inlet.index({Slice(), 3-1}) - outlet.index({Slice(), 3-1})
    + inlet.index({Slice(), 5-1}) - outlet.index({Slice(), 5-1})
  );
}

void inlet(torch::Tensor& inlet, const torch::Tensor& outlet, double dp)
{
  double factor = 1.0;
  if (dp==0.0) factor = 0.0;
  auto c = factor*C(inlet, outlet, dp);
  //cout << "C=\n" << c <<endl;
  // cout << "inlet before\n" << inlet << endl;
  // cout << "and the outlet\n" << outlet << endl;
  inlet.index({Slice(), 2-1}) = outlet.index({Slice(), 2-1}) + c;
  inlet.index({Slice(), 6-1}) = outlet.index({Slice(), 6-1}) + 0.25*c;
  inlet.index({Slice(), 9-1}) = outlet.index({Slice(), 9-1}) + 0.25*c;
  // cout << "inlet after\n" << inlet << endl;
}

void outlet(const torch::Tensor& inlet, torch::Tensor& outlet, double dp)
{
  double factor = 1.0;
  if (dp==0.0) factor = 0.0;
  auto c = factor*C(inlet, outlet, dp);
  //cout << "C=\n" << c <<endl;
  outlet.index({Slice(), 4-1}) = inlet.index({Slice(), 4-1}) - c;
  outlet.index({Slice(), 7-1}) = inlet.index({Slice(), 7-1}) - 0.25*c;
  outlet.index({Slice(), 8-1}) = inlet.index({Slice(), 8-1}) - 0.25*c;
}

}
