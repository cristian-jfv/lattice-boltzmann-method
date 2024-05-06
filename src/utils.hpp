#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <torch/torch.h>

namespace utils
{

bool continue_execution();

typedef std::initializer_list<at::indexing::TensorIndex> indices;

void store_tensor(const torch::Tensor& source, torch::Tensor& target, int i);

template<typename T>
void print(const T& t, const char& end='\n')
{
  std::cout << t << end;
}

template<typename T>
void print(const std::string& name, const T& t, const char& end='\n')
{
  std::cout << name << "=" << t << end;
}
}

#endif
