#pragma once
#include <torch/extension.h>

#define BLOCK_SIZE 32    // 8x8, limit 1024: 32x32, 512: 24x24, 256: 16x16, 128: 10x10

// torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b);

// torch::Tensor pcg(
//     torch::Tensor i_lists, 
//     torch::Tensor j_lists,
//     torch::Tensor boson, 
//     torch::Tensor psi_u, 
//     torch::Tensor precon,
//     int64_t Lx, int64_t Ltau, int64_t max_iter, double rtol); // precon is a torch.sparse.csr tensor

torch::Tensor precon_vec(
    const torch::Tensor& d_r,        // [bs, Ltau * Vs] complex64
    const torch::Tensor& precon,     // [Ltau * Vs, Ltau * Vs] complex64, sparse_csr
    int64_t Lx); // Returns a tensor after applying preconditioning

torch::Tensor mhm_vec(
    const torch::Tensor& boson,   // [bs, Ltau * Vs * 2] float32
    const torch::Tensor& vec,     // [bs, Ltau * Vs] complex64
    const int64_t Lx,
    const float dtau,
    const int64_t block_size_x,
    const int64_t block_size_y);
    
torch::Tensor mh_vec(
    const torch::Tensor& boson,   // [bs, Ltau * Vs * 2] float32
    const torch::Tensor& vec,     // [bs, Ltau * Vs] complex64
    const int64_t Lx,
    const float dtau,
    const int64_t block_size_x,
    const int64_t block_size_y);

torch::Tensor b_vec_per_tau(
    const torch::Tensor& boson,   // [Ltau * Vs * 2] float32
    const torch::Tensor& vec,     // [Vs] complex64
    const int64_t Lx,
    const float dtau,
    const bool interm_out_bool,
    const int64_t block_size_x,
    const int64_t block_size_y); 