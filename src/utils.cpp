#include <iostream>
#include "utils.hpp"

namespace utils
{

bool continue_execution()
{
  char choice{'a'};
  while (true)
  {
    std::cout << "\nDo you want to continue (y/n)? ";
    std::cin >> choice;
    if (choice == 'y') return true;
    else if (choice == 'n') return false;

    std::cout << "Invalid input. Please enter 'y' or 'n'." << std::endl;
  }
}

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
