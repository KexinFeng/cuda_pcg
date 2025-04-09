#include <torch/torch.h>
#include <iostream>
#include "cuda_pcg.h" // Include the header file for the pcg function

int main() {
    // Initialize demo inputs
    int Lx = 2, Ltau = 40, max_iter = 50;
    double tol = 1e-6;

    // Example tensor inputs
    auto i_lists = torch::randint(0, 10, {4, 10}, torch::kInt32); // Random indices
    auto j_lists = torch::randint(0, 10, {4, 10}, torch::kInt32); // Random indices
    auto boson = torch::randn({10}, torch::kDouble);              // Random boson tensor
    auto psi_u = torch::randn({10, 1}, torch::kComplexDouble);    // Random psi_u tensor with complex128 type
    auto indices = torch::tensor({{0, 0, 1, 2, 2}, {0, 1, 2, 0, 1}}, torch::kInt64); // Row and column indices
    auto values = torch::randn({5}, torch::kDouble);                                 // Random values
    std::vector<int64_t> size = {10, 10};                                            // Size of the sparse matrix
    auto precon = torch::sparse_coo_tensor(indices, values, size).to_sparse_csr();   // Convert to CSR format

    // Call the pcg function
    torch::Tensor result = pcg(i_lists, j_lists, boson, psi_u, precon, Lx, Ltau, max_iter, tol);

    // Print the result
    std::cout << "PCG Result: " << result << std::endl;

    return 0;
}
