#include <iostream>
#include <ostream>
#include <torch/serialize.h>
#include <torch/torch.h>
#include "params.hpp"
#include "solver.hpp"
#include "domain.hpp"

using namespace std;

void store_tensor(const torch::Tensor& source, torch::Tensor& target, int i)
{
  using torch::indexing::Slice;
    // Save results
    target.index({Slice(), Slice(), Slice(), i}) =
      source.index({Slice(), Slice(), Slice()}).clone().detach();
}

int main()
{
  using torch::Tensor;
  using torch::indexing::Slice;
  using torch::indexing::None;

  cout << "\nProgram starts" << endl;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()){
    device = torch::kCUDA;
    std::cout << "CUDA is available, device=" << device << std::endl;
  }

  // Read flow parameters
  /*dimensional_params 
  dimensionless_params ;
  H = 200;
  W = 200;
  tau = 0.95;
  cout << << endl;
  cout << << endl;*/
  const double rho_0 = 1.0;
  const double p_0 = 1.0;
  const double tau = 0.95;
  const double eps = 1.0/tau;
  const int H = 50;
  const int W = 200;
  const int T = 10000; // TODO: Calculate number of time steps required
  const double dp = 0.00108; //0.081; //12.0*rho_0*nu/(H*H)*U * double(H);
  cout << "dp=" << dp << endl;

  // Initialize matrices
  Tensor f_curr = torch::zeros({H,W,9});
  Tensor f_next = torch::zeros_like(f_curr);
  Tensor f_equi = torch::zeros_like(f_curr);
  Tensor u_tens = torch::zeros({H,W,2});
  Tensor p_tens = torch::zeros({H,W,1});
  if (false && torch::cuda::is_available())
  {
    cout << "Sending tensors to: " << device << endl;
    f_curr.to(device);
    f_next.to(device);
    f_equi.to(device);
    u_tens.to(device);
    p_tens.to(device);
  }

  initialize(rho_0, p_0, f_curr, u_tens, p_tens);
  // Create views for the domain boundaries
  //auto left = domain::left_boundary(f_next);
  //cout << "left.sizes=" << left.sizes() << endl;
  //auto right = domain::right_boundary(f_next);
  //auto top = domain::top_boundary(f_next);
  //cout << "top.sizes=" << top.sizes() << endl;
  //auto bottom = domain::bottom_boundary(f_next);

  // Results storage
  Tensor ux = torch::zeros({H, W, T});
  Tensor uy = torch::zeros({H, W, T});
  Tensor ps = torch::zeros({H, W, T});
  Tensor fs = torch::zeros({H, W, 9, T});

  // Main loop
  cout << "===================================================================" << endl;
  cout << "Main loop starts" << endl;
  cout << "===================================================================" << endl;

  for (int i=0; i<T; i++)
  {
    cout << i << "\t\r" << flush;
    //cout << "ux=\n" << u_tens.index({Slice(), Slice(), 0}) << endl;
    //cout << "uy=\n" << u_tens.index({Slice(), Slice(), 1}) << endl;
    //cout << "u=\n" << u_tens << endl;
    //cout << "ps=\n" << p_tens << endl;
    //cout << "f=\n" << f_curr << endl;
    //cout << "===================================================================" << endl;
    // Save results
    ux.index({Slice(), Slice(), i}) = u_tens.index({Slice(), Slice(), 0}).clone().detach();
    uy.index({Slice(), Slice(), i}) = u_tens.index({Slice(), Slice(), 1}).clone().detach();
    ps.index({Slice(), Slice(), i}) = p_tens.index({Slice(), Slice(), 0}).clone().detach();

    f_eq(f_equi, u_tens, p_tens);
    //f_step(f_next, f_curr, f_equi, eps);
    // cout << "f_next before boundary conditions\n" << f_next << endl;
    auto temp = (f_curr - eps*(f_curr - f_equi)).clone().detach();
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

    // Create views for the domain boundaries
    auto top = domain::top_boundary(f_next);
    auto bottom = domain::bottom_boundary(f_next);
    auto left = domain::left_boundary(f_next);
    auto right = domain::right_boundary(f_next);
    // Enforce boundary conditions
    // No slip at TOP
    domain::no_slip(top, domain::interface::fluid_to_wall);
    // No slip at BOTTOM
    domain::no_slip(bottom, domain::interface::wall_to_fluid);
    // Inlet at LEFT
    domain::inlet(left, right, dp);
    // Outlet at RIGHT
    domain::outlet(left, right, dp);
    f_next.index({0,0,6-1}) = f_next.index({0,0,8-1}).clone().detach();
    f_next.index({0,-1,6-1}) = f_next.index({0,-1,8-1}).clone().detach();

    f_next.index({0,0,7-1}) = f_next.index({0,0,9-1}).clone().detach();
    f_next.index({0,-1,7-1}) = f_next.index({0,-1,9-1}).clone().detach();

    f_next.index({-1,0,8-1}) = f_next.index({-1,0,6-1}).clone().detach();
    f_next.index({-1,-1,8-1}) = f_next.index({-1,-1,6-1}).clone().detach();

    f_next.index({-1,0,9-1}) = f_next.index({-1,0,7-1}).clone().detach();
    f_next.index({-1,-1,9-1}) = f_next.index({-1,-1,7-1}).clone().detach();
    store_tensor(f_next, fs, i);

    f_curr = f_next.clone().detach();

    u(u_tens, f_curr);
    p(p_tens, f_curr);
  }
  // cout << "u=\n" << u_tens << endl;
  // cout << "ps=\n" << p_tens << endl;
  // cout << "f_curr=\n" << f_curr << endl;
  // cout << "f_next=\n" << f_next << endl;
  // cout << "===================================================================" << endl;


  // Save results
  cout << "Saving results" << endl;
  torch::save(ux, "ux.pt");
  torch::save(uy, "uy.pt");
  torch::save(ps, "ps.pt");
  torch::save(fs, "fs.pt");

  cout << "Program ends" << endl;
  return 0;
}
