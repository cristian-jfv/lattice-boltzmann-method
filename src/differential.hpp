#ifndef DIFFERENTIAL_HPP
#define DIFFERENTIAL_HPP

#include <torch/torch.h>

class differential
{
  private:
  const torch::Tensor xi = (1.0/5040.0)*torch::tensor(
    {{ 1.0,  32.0,  84.0,  32.0,  1.0},
     {32.0, 448.0, 960.0, 448.0, 32.0},
     {84.0, 960.0,   0.0, 960.0, 84.0},
     {32.0, 448.0, 960.0, 448.0, 32.0},
     { 1.0,  32.0,  84.0,  32.0,  1.0}},
    torch::TensorOptions()
    .dtype(torch::kDouble)
    .device(torch::kCUDA)
    .requires_grad(false));

  const torch::Tensor kernel_partial_y = torch::tensor(
    {{-2.0, -1.0, 0.0, 1.0, 2.0},
     {-2.0, -1.0, 0.0, 1.0, 2.0},
     {-2.0, -1.0, 0.0, 1.0, 2.0},
     {-2.0, -1.0, 0.0, 1.0, 2.0},
     {-2.0, -1.0, 0.0, 1.0, 2.0}},
    torch::TensorOptions()
    .dtype(torch::kDouble)
    .device(torch::kCUDA)
    .requires_grad(false));

  const torch::Tensor kernel_partial_x = -torch::tensor(
    {{ 2.0,  2.0,  2.0,  2.0,  2.0},
     { 1.0,  1.0,  1.0,  1.0,  1.0},
     { 0.0,  0.0,  0.0,  0.0,  0.0},
     {-1.0, -1.0, -1.0, -1.0, -1.0},
     {-2.0, -2.0, -2.0, -2.0, -2.0}},
    torch::TensorOptions()
    .dtype(torch::kDouble)
    .device(torch::kCUDA)
    .requires_grad(false));

  // Convolution operators for partial derivatives
  torch::nn::Conv2d partial_x = nullptr;
  torch::nn::Conv2d partial_y = nullptr;
  torch::nn::Conv2d initialize_convolution(const torch::Tensor &kernel);

  public:
  differential();
  torch::Tensor x(const torch::Tensor& psi);
  torch::Tensor y(const torch::Tensor& psi);
  void grad(torch::Tensor& ans, const torch::Tensor& psi);

};

#endif
