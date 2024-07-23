#ifndef ULBM_HPP
#define ULBM_HPP

#include <c10/core/DeviceType.h>
#include <torch/torch.h>

namespace ulbm
{

namespace d2q9
{
class kbc
{
  public:
  torch::Tensor iequi_f; // Inverse multiplicative of equi_f
  torch::Tensor coll_f;
  torch::Tensor adve_f;
  torch::Tensor m0; // Zero-order moment of the dist. function
  torch::Tensor m1; // First-order moment of the dist. function
  void collide();
  void advect();
  kbc(int R, int C, double s2);
  void eval_equilibrium(torch::Tensor& equi_f);
  private:
  const double s2;
  const double is2;
  const double cs2 = 1.0/3.0;
  const double cs4 = 1.0/9.0;
  const torch::Tensor inv_M = torch::tensor(
    {
        {1.0,    0.0,    0.0,   -1.0,    0.0,   0.0,   0.0,   0.0,   1.0},
        {0.0,    0.5,    0.0,   0.25,   0.25,   0.0,   0.0,  -0.5,  -0.5},
        {0.0,    0.0,    0.5,   0.25,  -0.25,   0.0,  -0.5,   0.0,  -0.5},
        {0.0,   -0.5,    0.0,   0.25,   0.25,   0.0,   0.0,   0.5,  -0.5},
        {0.0,    0.0,   -0.5,   0.25,  -0.25,   0.0,   0.5,   0.0,  -0.5},
        {0.0,    0.0,    0.0,    0.0,    0.0,  0.25,  0.25,  0.25,  0.25},
        {0.0,    0.0,    0.0,    0.0,    0.0, -0.25,  0.25, -0.25,  0.25},
        {0.0,    0.0,    0.0,    0.0,    0.0,  0.25, -0.25, -0.25,  0.25},
        {0.0,    0.0,    0.0,    0.0,    0.0, -0.25, -0.25,  0.25,  0.25},
    }, torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));
  const torch::Tensor minus_inv_M = -inv_M;
  // inv_N is not defined, instead explicit algebraic expressions are used
  // to perform the matrix multiplication
  // const torch::Tensor inv_N;
  torch::Tensor S;
  torch::Tensor gamma;
  torch::Tensor delta_s;
  torch::Tensor delta_h;
  // Central momenta for the collision step
  torch::Tensor inter_coll_f; // store for intermediate results for the coll. step
  // const torch::Tensor& icf0 = inter_coll_f.index({torch::indexing::Ellipsis,0});
  // const torch::Tensor& icf1 = inter_coll_f.index({torch::indexing::Ellipsis,1});
  // const torch::Tensor& icf2 = inter_coll_f.index({torch::indexing::Ellipsis,2});
  // const torch::Tensor& icf3 = inter_coll_f.index({torch::indexing::Ellipsis,3});
  // const torch::Tensor& icf4 = inter_coll_f.index({torch::indexing::Ellipsis,4});
  // const torch::Tensor& icf5 = inter_coll_f.index({torch::indexing::Ellipsis,5});
  // const torch::Tensor& icf6 = inter_coll_f.index({torch::indexing::Ellipsis,6});
  // const torch::Tensor& icf7 = inter_coll_f.index({torch::indexing::Ellipsis,7});
  // const torch::Tensor& icf8 = inter_coll_f.index({torch::indexing::Ellipsis,8});
  torch::Tensor cT; // this is \tilde{T}
  // const torch::Tensor& cT0 = cT.index({torch::indexing::Ellipsis,0});
  // const torch::Tensor& cT1 = cT.index({torch::indexing::Ellipsis,1});
  // const torch::Tensor& cT2 = cT.index({torch::indexing::Ellipsis,2});
  // const torch::Tensor& cT3 = cT.index({torch::indexing::Ellipsis,3});
  // const torch::Tensor& cT4 = cT.index({torch::indexing::Ellipsis,4});
  // const torch::Tensor& cT5 = cT.index({torch::indexing::Ellipsis,5});
  // const torch::Tensor& cT6 = cT.index({torch::indexing::Ellipsis,6});
  // const torch::Tensor& cT7 = cT.index({torch::indexing::Ellipsis,7});
  // const torch::Tensor& cT8 = cT.index({torch::indexing::Ellipsis,8});

  torch::Tensor Ts; // raw moment, used to calculate gamma
  torch::Tensor Th; // raw moment, used to calculate gamma
  // const torch::Tensor& ux = m1.index({torch::indexing::Ellipsis,0});
  // const torch::Tensor& uy = m1.index({torch::indexing::Ellipsis,1});
  torch::Tensor ux2;
  torch::Tensor uy2;
  torch::Tensor cmx;
  torch::Tensor cmy;
  torch::Tensor cmx2;
  torch::Tensor cmy2;
  torch::Tensor cm_buffer;
  void eval_central_momenta();
  void eval_s_matrix();
  void eval_gamma();
  void eval_m1_components();
  void eval_delta_s();
  void eval_delta_h();
  void eval_iequilibrium();
};
}

}

#endif
