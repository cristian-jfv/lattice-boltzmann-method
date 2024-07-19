#ifndef DOMAIN_HPP
#define DOMAIN_HPP
#include <torch/torch.h>

struct domain
{
  torch::Tensor adve_f;
  torch::Tensor equi_f;
  torch::Tensor coll_f;
  torch::Tensor m_0; // typically density
  torch::Tensor m_1; // typically momentum
  domain(int R, int C, int Q=9);
};

#endif
