#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/complex.h>
#include "cuda_pcg.h"


// (Ltau / BLOCK_WIDTH) blocks; 
// halo region size is 6, equal to PAD;
// BLOCK_WIDTH is the number of output pixels per block. 
// TILE_SIZE = BLOCK_WIDTH + 2*PAD
// KENEL_SIZE is equal to BLOCK_WIDTH + 2*PAD in this case
// across the batch, kernel is shared.
#define BLOCK_WIDTH 32
#define PAD 6
#define TILE_SIZE (BLOCK_WIDTH + 2 * PAD)
#define KERNEL_SIZE (BLOCK_WIDTH + 2 * PAD)
#define NUM_ENTRY_PER_ROW 13
#define NUM_STREAMS 32  // Reuse a fixed pool of streams; might needs to increase for Lx > 24

namespace cuda_pcg {
template<typename scalar_t>
__global__ void precon_vec_kernel(
    const scalar_t* __restrict__ d_r,  // [bs, Ltau * Vs] complex64
    const int64_t* __restrict__ precon_crow, // [num_rows + 1]
    const int64_t* __restrict__ precon_col, // [nnz], Ltau * Vs
    const scalar_t* __restrict__ precon_val, // [nnz], complex64, Ltau * Vs
    scalar_t* __restrict__ out,         // [bs, Ltau * Vs] complex64
    const int Lx,  // typically Lx^2 = 10x10 = 100, up to 24x24 = 576
    const int Ltau, // typically 400, up to 24x40 = 960
    const int bs) 
{
    // Allocate shared mem for stencil and input_tile
    // __shared__ scalar_t s_crow[Ltau * Lx * Lx + 1];
    __shared__ scalar_t s_col[KERNEL_SIZE][NUM_ENTRY_PER_ROW];
    __shared__ scalar_t s_val[KERNEL_SIZE][NUM_ENTRY_PER_ROW];
    __shared__ scalar_t s_input_tile[TILE_SIZE];

    int tx = threadIdx.x; 
    int idx_tau = blockIdx.x * blockDim.x + tx;  // global temporal idx: [blockIdx.x, threadIdx.x]
    int idx_site = blockIdx.y;

    int stride_vs = Lx * Lx;
    
    // Load the input into shared memory
    s_input_tile[PAD + tx] = d_r[idx_tau * stride_vs + idx_site]; 

    if (tx < PAD) { // Left halo
        int idx_tau_pad = (idx_tau - PAD) % Ltau;  // left shift each idx_tau by PAD
        s_input_tile[tx] = d_r[idx_tau_pad * stride_vs + idx_site]; 
    }
    if (tx >= blockDim.x - PAD) { // Right halo; backward count by PAD
        int idx_tau_pad = (idx_tau + PAD) % Ltau;  // right shift each idx_tau by PAD
        s_input_tile[tx + 2*PAD] = /* starting at blockDim.x + PAD when tx = blockDim.x - PAD */
        d_r[idx_tau_pad * stride_vs + idx_site];
    }

    // Load stencil into shared memory
    int row_start = precon_crow[idx_tau * stride_vs + idx_site];
    int row_size = precon_crow[idx_tau * stride_vs + idx_site + 1] - row_start;  // ~ Num_ENTRY_PER_ROW
    for (int i = 0; i < row_size; ++i) {
        // int tau_shift = i - Num_ENTRY_PER_ROW / 2;  // [-6, ..., 0, ..., 6]
        // s_col[PAD + tx, i] = precon_col[precon_crow[idx_tau * stride_vs + idx_site] + (idx_tau + tau_shift)%Ltau * stride_vs + idx_site]  // [Ltau * Vs], tx->row of shared mat, i->col of shared mat, 
        // `(idx_tau + tau_shift)%Ltau * stride_vs` would be used if a row were not compressed.
        s_col[PAD + tx, i] = precon_col[row_start + i];  // [Ltau * Vs], tx->row of shared mat, i->col of shared mat
        s_val[PAD + tx, i] = precon_val[row_start + i];  // [Ltau * Vs], tx->row of shared mat, i->col of shared mat
    }

    if (tx < PAD) { // Left halo
        int idx_tau_pad = (idx_tau - PAD) % Ltau;
        int row_start_pad = precon_crow[idx_tau_pad * stride_vs + idx_site];
        int row_size_pad = precon_crow[idx_tau_pad * stride_vs + idx_site + 1] - row_start_pad; 
        for (int i = 0; i < row_size_pad; ++i) {
            s_col[tx, i] = precon_col[row_start_pad + i]; 
            s_val[tx, i] = precon_val[row_start_pad + i];  
        }
    }
    if (tx >= blockDim.x - PAD) { // Right halo
        int idx_tau_pad = (idx_tau + PAD) % Ltau;
        int row_start_pad = precon_crow[idx_tau_pad * stride_vs + idx_site];
        int row_size_pad = precon_crow[idx_tau_pad * stride_vs + idx_site + 1] - row_start_pad; 
        for (int i = 0; i < row_size_pad; ++i) {
            s_col[tx + 2*PAD, i] = precon_col[row_start_pad + i];  // [Ltau * Vs], tx->row of shared mat, i->col of shared mat
            s_val[tx + 2*PAD, i] = precon_val[row_start_pad + i];  // [Ltau * Vs], tx->row of shared mat, i->col of shared mat
        }
    }
    __syncthreads();

    // Compute the output
    // ... 

    if (tx == 0 && blockIdx.x == 0) {
        printf("Debug: Code reached here. BlockIdx.x: %d, ThreadIdx.x: %d\n", blockIdx.x, tx);
    }
}
} // namespace cuda_pcg

void precon_vec(
    const torch::Tensor& d_r,        // [bs, Ltau * Vs] complex64
    const torch::Tensor& precon,     // [Ltau * Vs, Ltau * Vs] complex64, sparse_csr
    torch::Tensor& out,
    int Lx) 
{
    TORCH_CHECK(d_r.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(precon.is_cuda(), "Kernel must be a CUDA tensor");

    // auto out = torch::empty_like(d_r);
    auto precon_crow = precon.crow_indices();
    auto precon_col = precon.col_indices();
    auto precon_val = precon.values();
    
    auto bs = d_r.size(0);
    auto Vs = Lx * Lx;  // typically 10x10 = 100, up to 24x24 = 576
    auto Ltau = precon.size(0) / Vs;  // typically 400, up to 24x40 = 960
    
    dim3 block = {BLOCK_WIDTH};  
    auto num_blocks = (Ltau + BLOCK_WIDTH - 1) / BLOCK_WIDTH; 
    dim3 grid = {num_blocks, Vs};

    using scalar_t = cuFloatComplex;  // Default declaration
    if (d_r.dtype() == at::ScalarType::ComplexFloat) {
        using scalar_t = cuFloatComplex; 
    } else if (d_r.dtype() == at::ScalarType::ComplexDouble) {
        using scalar_t = cuDoubleComplex;
    } else {
        throw std::invalid_argument("Unsupported data type");
    }

    int dyn_shared_mem = (sizeof(int64_t) * KERNEL_SIZE * NUM_ENTRY_PER_ROW  // s_col
                        + sizeof(scalar_t) * KERNEL_SIZE * NUM_ENTRY_PER_ROW  // s_val
                        + sizeof(scalar_t) * TILE_SIZE) // s_input_tile
                        * block.x;           

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cuda_pcg::precon_vec_kernel<scalar_t><<<grid, block, dyn_shared_mem, stream>>>(
        d_r.data_ptr<scalar_t>(), 
        precon_crow.data_ptr<int64_t>(), 
        precon_col.data_ptr<int64_t>(), 
        precon_val.data_ptr<scalar_t>(), 
        out.data_ptr<scalar_t>(), 
        Lx, Ltau, bs);

    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA stream synchronization failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel execution failed");
        return; 
    }
}
