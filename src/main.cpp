#include <iostream>
#include <ostream>
#include <torch/torch.h>
#include "params.hpp"
#include "solver.hpp"

using namespace std;

int main()
{
	using torch::Tensor;

	cout << "Program starts" << endl;
	// Read flow parameters
	dimensional_params dap;
	dimensionless_params dlp{dap};
	dlp.H = 500;
	dlp.W = 500;
	cout << dap << endl;
	cout << dlp << endl;
	int T = 123; // TODO: Calculate number of time steps required

	// Initialize matrices
	Tensor f = torch::zeros({dlp.H,dlp.W,9});
	Tensor u = torch::zeros({dlp.H,dlp.W,2});
	Tensor p = torch::zeros({dlp.H,dlp.W,1});

	initialize(1.0, 1.0, f, u, p);

	// Main loop
	for (int i=0; i<T; i++)
	{
		// f_step
		// recalculate u and p
	}

	// Save results
	cout << "Program ends" << endl;
	return 0;
}
