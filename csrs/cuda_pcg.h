#pragma once
#include <torch/extension.h>


torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b);

torch::Tensor pcg(
    torch::Tensor i_lists, 
    torch::Tensor j_lists,
    torch::Tensor boson, 
    torch::Tensor psi_u, 
    torch::Tensor precon,
    int Lx, int Ltau, int max_iter, double rtol); // precon is a torch.sparse.csr tensor

torch::Tensor precon_vec(
    const torch::Tensor& d_r,        // [bs, Ltau * Vs] complex64
    const torch::Tensor& precon,     // [Ltau * Vs, Ltau * Vs] complex64, sparse_csr
    int Lx); // Returns a tensor after applying preconditioning

