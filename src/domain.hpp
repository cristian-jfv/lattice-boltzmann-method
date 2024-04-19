#ifndef DOMAIN_HPP
#define DOMAIN_HPP
#include <torch/torch.h>

namespace domain
{

enum interface{wall_to_fluid, fluid_to_wall};

torch::Tensor left_boundary(torch::Tensor& domain);

torch::Tensor right_boundary(torch::Tensor& domain);

torch::Tensor top_boundary(torch::Tensor& domain);

torch::Tensor bottom_boundary(torch::Tensor& domain);

void no_slip(torch::Tensor& boundary, interface itf);

void specular(torch::Tensor& boundary, interface itf);

void inlet(torch::Tensor& inlet, const torch::Tensor& outlet, double dp);

void outlet(const torch::Tensor& inlet, torch::Tensor& outlet, double dp);

}
#endif
