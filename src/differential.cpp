#include "differential.hpp"

torch::nn::Conv2d differential::initialize_convolution(const torch::Tensor &kernel)
{
  auto conv_options = torch::nn::Conv2dOptions(
    /*in_channels=*/1, /*out_channels=*/1, /*kernel_size=*/5)
                               .padding({2,2})
                               .padding_mode(torch::kReplicate)
                               .bias(false);

  torch::nn::Conv2d ans(conv_options);
  ans->weight = (xi*kernel).reshape({1, 1, 5, 5}).clone().detach();
  ans->bias = torch::tensor({0.0},torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));
  return ans;
}

differential::differential()
{
  partial_x = initialize_convolution(kernel_partial_x);
  partial_y = initialize_convolution(kernel_partial_y);
}

torch::Tensor differential::x(const torch::Tensor &psi)
{
  return partial_x->forward(psi.unsqueeze(0).unsqueeze(0).squeeze(-1))
  .squeeze(0).squeeze(0).clone().detach();
}

torch::Tensor differential::y(const torch::Tensor &psi)
{
  return partial_y->forward(psi.unsqueeze(0).unsqueeze(0).squeeze(-1))
  .squeeze(0).squeeze(0).clone().detach();
}

void differential::grad(torch::Tensor &ans, const torch::Tensor &psi)
{
  ans.index({torch::indexing::Ellipsis,0}) = x(psi).clone().detach();
  ans.index({torch::indexing::Ellipsis,1}) = y(psi).clone().detach();
}
