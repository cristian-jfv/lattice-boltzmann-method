#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <torch/torch.h>

namespace utils
{

typedef std::initializer_list<at::indexing::TensorIndex> indices;

void store_tensor(const torch::Tensor& source, torch::Tensor& target, int i);

void print(const std::string& str);

template<typename T>
void print(const std::string& name, const T& t)
{
  std::cout << name << "=" << t << std::endl;
}
}

#endif
