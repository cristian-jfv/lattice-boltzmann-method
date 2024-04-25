#include <iostream>
#include "utils.hpp"

namespace utils
{
void store_tensor(const torch::Tensor& source, torch::Tensor& target, int i)
{
  using torch::indexing::Slice;
  using torch::indexing::Ellipsis;

    // Save results
    target.index({Ellipsis, i}) =
      source.index({Slice(), Slice(), Slice()}).clone().detach();
}

void print(const std::string& str)
{
  std::cout << str << std::endl;
}

}
