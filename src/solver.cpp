# include "solver.hpp"

namespace solver
{
using torch::indexing::Slice;
using torch::indexing::None;

const torch::Tensor E = torch::tensor(
  {4.0/ 9.0,
    1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0});
const torch::Tensor c = torch::tensor(
  {{0.0, 1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
   {0.0, 0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0}});

void initialize(double rho_0, double p_0,
                torch::Tensor &f,
                torch::Tensor &u,
                torch::Tensor &p)
{
  // Fill f, u, and p with initial values
  u =torch::zeros_like(u);
  p = (rho_0/3.0)*torch::ones_like(p);
  f_eq(f, u, p);
}

void f_eq(torch::Tensor &f_eq,
          const torch::Tensor &u,
          const torch::Tensor &p)
{
  auto u_u = (u*u).sum_to_size(p.sizes());
  auto c_u = matmul(u, c);
  f_eq = mul(3.0*p + 3.0*c_u + 4.5*c_u.pow(2) - 1.5*u_u, E).clone().detach();
}

void f_step(torch::Tensor& f_next,
            const torch::Tensor& f_curr,
            const torch::Tensor& f_eq,
            double eps)
{
    auto temp = (f_curr - eps*(f_curr - f_eq)).clone().detach();
    // Advect
    // f1
    f_next.index({Slice(), Slice(), 1-1}) =
      temp.index({Slice(), Slice(), 1-1}).clone().detach();

    // f2
    f_next.index({Slice(), Slice(1,None), 2-1}) =
      temp.index({Slice(), Slice(0,-1), 2-1}).clone().detach();
    // f2 carry
    f_next.index({Slice(), 0, 2-1}) =
      f_curr.index({Slice(), 0, 2-1}).clone().detach();

    // f3
    f_next.index({Slice(0,-1), Slice(), 3-1}) =
      temp.index({Slice(1,None), Slice(), 3-1}).clone().detach();
    // f3 carry
    f_next.index({-1, Slice(), 3-1}) =
      f_curr.index({-1, Slice(), 3-1}).clone().detach();

    // f4
    f_next.index({Slice(), Slice(0,-1), 4-1}) =
      temp.index({Slice(), Slice(1,None), 4-1}).clone().detach();
    // f4 carry
    f_next.index({Slice(), -1, 4-1}) =
      f_curr.index({Slice(), -1, 4-1}).clone().detach();

    // f5
    f_next.index({Slice(1,None), Slice(), 5-1}) =
      temp.index({Slice(0,-1), Slice(), 5-1}).clone().detach();
    // f5 carry
    f_next.index({0, Slice(), 5-1}) =
      f_curr.index({0, Slice(), 5-1}).clone().detach();

    // f6
    f_next.index({Slice(0,-1), Slice(1,None), 6-1}) =
      temp.index({Slice(1,None), Slice(0,-1), 6-1}).clone().detach();
    // f6 carry
    f_next.index({Slice(), 0, 6-1}) =
      f_curr.index({Slice(), 0, 6-1}).clone().detach();
    f_next.index({-1, Slice(), 6-1}) =
      f_curr.index({-1, Slice(), 6-1}).clone().detach();


    // f7
    f_next.index({Slice(0,-1), Slice(0,-1), 7-1}) =
      temp.index({Slice(1,None), Slice(1,None), 7-1}).clone().detach();
    f_next.index({Slice(), -1, 7-1}) =
      f_curr.index({Slice(), -1, 7-1}).clone().detach();
    f_next.index({-1, Slice(), 7-1}) =
      f_curr.index({-1, Slice(), 7-1}).clone().detach();

    // f8
    f_next.index({Slice(1,None), Slice(0,-1), 8-1}) =
      temp.index({Slice(0,-1), Slice(1,None), 8-1}).clone().detach();
    f_next.index({Slice(), -1, 8-1}) =
      f_curr.index({Slice(), -1, 8-1}).clone().detach();
    f_next.index({0, Slice(), 8-1}) =
      f_curr.index({0, Slice(), 8-1}).clone().detach();

    // f9
    f_next.index({Slice(1,None), Slice(1,None), 9-1}) =
      temp.index({Slice(0,-1), Slice(0,-1), 9-1}).clone().detach();
    f_next.index({0, Slice(), 9-1}) =
      f_curr.index({0, Slice(), 9-1}).clone().detach();
    f_next.index({Slice(), 0, 9-1}) =
      f_curr.index({Slice(), 0, 9-1}).clone().detach();
}

void recover_corners(torch::Tensor& f_next)
{
    f_next.index({0,0,6-1}) = f_next.index({0,0,8-1}).clone().detach();
    f_next.index({0,-1,6-1}) = f_next.index({0,-1,8-1}).clone().detach();

    f_next.index({0,0,7-1}) = f_next.index({0,0,9-1}).clone().detach();
    f_next.index({0,-1,7-1}) = f_next.index({0,-1,9-1}).clone().detach();

    f_next.index({-1,0,8-1}) = f_next.index({-1,0,6-1}).clone().detach();
    f_next.index({-1,-1,8-1}) = f_next.index({-1,-1,6-1}).clone().detach();

    f_next.index({-1,0,9-1}) = f_next.index({-1,0,7-1}).clone().detach();
    f_next.index({-1,-1,9-1}) = f_next.index({-1,-1,7-1}).clone().detach();
}

void p(torch::Tensor& p,
       const torch::Tensor& f)
{
  p = f.sum_to_size(p.sizes()).clone().detach()/3.0;
}

void u(torch::Tensor& u,
       const torch::Tensor& f)
{
  u = matmul(f, c.transpose(0,1)).clone().detach();
}
}
