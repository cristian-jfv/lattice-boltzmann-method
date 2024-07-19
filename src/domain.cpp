#include "domain.hpp"

domain::domain(int R, int C, int Q)
{
  adve_f = torch::zeros({R,C,Q}, torch::kCUDA);
  equi_f = torch::zeros({R,C,Q}, torch::kCUDA);
  coll_f = torch::zeros({R,C,Q}, torch::kCUDA);
  m_0 = torch::zeros({R,C,1}, torch::kCUDA);
  m_1 = torch::zeros({R,C,2}, torch::kCUDA);
}
