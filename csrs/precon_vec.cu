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
    // __shared__ int64_t s_crow[Ltau * Lx * Lx + 1];
    __shared__ int64_t s_col[KERNEL_SIZE][NUM_ENTRY_PER_ROW];
    __shared__ scalar_t s_val[KERNEL_SIZE][NUM_ENTRY_PER_ROW];
    __shared__ scalar_t s_input_tile[TILE_SIZE];

    int tx = threadIdx.x; 
    int idx_tau = blockIdx.y * blockDim.y + tx;  // global temporal idx: [blockIdx.y, threadIdx.x]
    if (idx_tau >= Ltau) return;

    int idx_site = blockIdx.x;
    int stride_vs = Lx * Lx;
    if (idx_site >= stride_vs) return;

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
        s_col[PAD + tx][i] = precon_col[row_start + i];  // [Ltau * Vs], tx->row of shared mat, i->col of shared mat
        s_val[PAD + tx][i] = precon_val[row_start + i];  // [Ltau * Vs], tx->row of shared mat, i->col of shared mat
    }

    if (tx < PAD) { // Left halo
        int idx_tau_pad = (idx_tau - PAD) % Ltau;
        int row_start_pad = precon_crow[idx_tau_pad * stride_vs + idx_site];
        int row_size_pad = precon_crow[idx_tau_pad * stride_vs + idx_site + 1] - row_start_pad; 
        for (int i = 0; i < row_size_pad; ++i) {
            s_col[tx][i] = precon_col[row_start_pad + i]; 
            s_val[tx][i] = precon_val[row_start_pad + i];  
        }
    }
    if (tx >= blockDim.x - PAD) { // Right halo
        int idx_tau_pad = (idx_tau + PAD) % Ltau;
        int row_start_pad = precon_crow[idx_tau_pad * stride_vs + idx_site];
        int row_size_pad = precon_crow[idx_tau_pad * stride_vs + idx_site + 1] - row_start_pad; 
        for (int i = 0; i < row_size_pad; ++i) {
            s_col[tx + 2*PAD][i] = precon_col[row_start_pad + i];  // [Ltau * Vs], tx->row of shared mat, i->col of shared mat
            s_val[tx + 2*PAD][i] = precon_val[row_start_pad + i];  // [Ltau * Vs], tx->row of shared mat, i->col of shared mat
        }
    }
    __syncthreads();

    if (tx == 0 && blockIdx.y == 0) {
        printf("Debug: Code reached here. BlockIdx.y: %d, ThreadIdx.x: %d\n", blockIdx.y, tx);
        printf("BlockIdx.x: %d\n", blockIdx.x);
        printf("s_col[0]: ");
        for (int i = 0; i < NUM_ENTRY_PER_ROW; ++i) {
            printf("%lld ", s_col[0][i]);
        }
        printf("\ns_val[0]: ");
        for (int i = 0; i < NUM_ENTRY_PER_ROW; ++i) {
            printf("(%f, %f) ", cuCrealf(s_val[0][i]), cuCimagf(s_val[0][i]));
        }
        printf("\ns_input_tile: ");
        for (int i = 0; i < TILE_SIZE; ++i) {
            printf("(%f, %f) ", cuCrealf(s_input_tile[i]), cuCimagf(s_input_tile[i]));
        }
        printf("\n");
    }

    // Compute the output
    // ... 
    
}
} // namespace cuda_pcg

torch::Tensor precon_vec(
    const torch::Tensor& d_r,        // [bs, Ltau * Vs] complex64
    const torch::Tensor& precon,     // [Ltau * Vs, Ltau * Vs] complex64, sparse_csr
    int Lx) 
{
    TORCH_CHECK(d_r.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(precon.is_cuda(), "Kernel must be a CUDA tensor");
    TORCH_CHECK(d_r.scalar_type() == at::ScalarType::ComplexFloat, "Input tensor must be of type ComplexFloat");

    // auto out = torch::empty_like(d_r);
    auto precon_crow = precon.crow_indices();
    auto precon_col = precon.col_indices();
    auto precon_val = precon.values();

    auto out = torch::empty_like(d_r);
    
    auto bs = d_r.size(0);
    auto Vs = Lx * Lx;  // typically 10x10 = 100, up to 24x24 = 576
    auto Ltau = precon.size(0) / Vs;  // typically 400, up to 24x40 = 960
    
    dim3 block = {BLOCK_WIDTH};  
    auto num_blocks = (Ltau + BLOCK_WIDTH - 1) / BLOCK_WIDTH; 
    dim3 grid = {Vs, num_blocks};

    // Verify that grid/block sizes are within device limits using:
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    // Check device pointer befor launch
    printf("d_r: %p, precon_val: %p\n", d_r.data_ptr(), precon_val.data_ptr());

    using scalar_t = cuFloatComplex;
    if (d_r.dtype() == at::ScalarType::ComplexFloat) {
        using scalar_t = cuFloatComplex; 
    } else if (d_r.dtype() == at::ScalarType::ComplexDouble) {
        using scalar_t = cuDoubleComplex;
    } else {
        throw std::invalid_argument("Unsupported data type");
    }

    int shared_mem = (sizeof(int64_t) * KERNEL_SIZE * NUM_ENTRY_PER_ROW  // s_col
                        + sizeof(scalar_t) * KERNEL_SIZE * NUM_ENTRY_PER_ROW  // s_val
                        + sizeof(scalar_t) * TILE_SIZE) // s_input_tile
                        * BLOCK_WIDTH;           

    printf("Shared memory requirement per block: %d bytes\n", shared_mem);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cuda_pcg::precon_vec_kernel<scalar_t><<<grid, block, 0, stream>>>(
        reinterpret_cast<scalar_t*>(d_r.data_ptr()), 
        precon_crow.data_ptr<int64_t>(), 
        precon_col.data_ptr<int64_t>(), 
        reinterpret_cast<scalar_t*>(precon_val.data_ptr()), 
        reinterpret_cast<scalar_t*>(out.data_ptr()), 
        Lx, Ltau, bs);

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(kernel_err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }

    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA stream synchronization failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel execution failed");
    }

    return out;
}
