#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/complex.h>
#include "cuda_pcg.h"
#include "utils.h"

// (Ltau / BLOCK_WIDTH) blocks; 
// halo region size is 6, equal to PAD;
// BLOCK_WIDTH is the number of output pixels per block. 
// TILE_SIZE = BLOCK_WIDTH + 2*PAD
// STENCIL_SIZE is equal to BLOCK_WIDTH in this case
// across the batch, stencil is shared.
#define BLOCK_WIDTH 64
#define PAD 6
#define TILE_SIZE (BLOCK_WIDTH + 2 * PAD)
#define STENCIL_SIZE BLOCK_WIDTH
#define NUM_ENTRY_PER_ROW 13 // PAD * 2 + 1
// #define NUM_STREAMS 32  // Reuse a fixed pool of streams; might needs to increase for Lx > 24

namespace cuda_pcg {
template<typename scalar_t>
__global__ void precon_vec_kernel(
    const scalar_t* __restrict__ d_r,  // [bs, Ltau * Vs] complex64
    const int64_t* __restrict__ precon_crow, // [num_rows + 1]
    const int64_t* __restrict__ precon_col, // [nnz], Ltau * Vs
    const scalar_t* __restrict__ precon_val, // [nnz], complex64, Ltau * Vs
    scalar_t* __restrict__ out,         // [bs, Ltau * Vs] complex64
    const int64_t Lx,  // typically Lx^2 = 10x10 = 100, up to 24x24 = 576
    const int64_t Ltau, // typically 400, up to 24x40 = 960
    const int64_t bs) 
{
    // Allocate shared mem for stencil and input_tile
    __shared__ scalar_t s_val[STENCIL_SIZE][NUM_ENTRY_PER_ROW];
    __shared__ int64_t s_col[STENCIL_SIZE][NUM_ENTRY_PER_ROW];
    extern __shared__ scalar_t s_input_tile_1d[]; // Declare shared memory as a 1D array
    scalar_t (*s_input_tile)[TILE_SIZE] = reinterpret_cast<scalar_t (*)[TILE_SIZE]>(s_input_tile_1d); // Cast to 2D array

    int64_t tx = threadIdx.x; 
    int64_t idx_tau = blockIdx.y * blockDim.x + tx;  // global temporal idx: [blockIdx.y, threadIdx.x]
    // BlockIdx.y ranges (num_blocks);
    // BlockDim.x == BLOCK_WIDTH, i.e. block_size_x
    int64_t row_size = 0;

    int64_t idx_site = blockIdx.x;
    int64_t stride_vs = Lx * Lx;
    int64_t stride_tau_vs = stride_vs * Ltau; 
    if (idx_site >= stride_vs) return;

    int64_t Vs = Lx * Lx; 

    int64_t max_tx_num = std::min(Ltau - blockIdx.y * blockDim.x, static_cast<int64_t>(blockDim.x));
    for (int64_t b = 0; b < bs; ++b) {
        // Load the input into shared memory
        if (tx < max_tx_num) {
            s_input_tile[b][PAD + tx] = d_r[b * stride_tau_vs + idx_tau * stride_vs + idx_site]; 
        }

        if (tx < PAD) { // Left halo
            // Possible that max_tx_num < PAD, but it's ok
            int64_t idx_tau_pad = mod(idx_tau - PAD, Ltau);  // left shift each idx_tau by PAD
            s_input_tile[b][tx] = d_r[b * stride_tau_vs + idx_tau_pad * stride_vs + idx_site]; 
        }

        if (tx >= blockDim.x - PAD) { // Right halo
            // Delta = tx - (blockDim.x - PAD)
            // idx_tau_pad = mod(blockIdx.y * blockDim.x + max_tx_num + Delta, Ltau)
            // d_r[idx_tau_pad * stride_vs + idx_site]
            // s_input_tile[PAD + max_tx_num + Delta]
            int64_t idx_tau_pad = mod(blockIdx.y * blockDim.x + max_tx_num + (tx - (blockDim.x - PAD)), Ltau); 
            s_input_tile[b][PAD + max_tx_num + (tx - (blockDim.x - PAD))] = d_r[b * stride_tau_vs + idx_tau_pad * stride_vs + idx_site]; 
            // s_input_tile[tx + 2*PAD] = d_r[mod(idx_tau + PAD, Ltau) * stride_vs + idx_site]; for non-corner case
        }
    }

    if (idx_tau < Ltau) {
        // Load stencil into shared memory
        int64_t row_start = precon_crow[idx_tau * stride_vs + idx_site];
        row_size = precon_crow[idx_tau * stride_vs + idx_site + 1] - row_start;  // ~ Num_ENTRY_PER_ROW
        for (int64_t i = 0; i < row_size; ++i) {
            s_val[tx][i] = precon_val[row_start + i];  // [Ltau * Vs], tx->row of shared mat, i->col of shared mat
            s_col[tx][i] = precon_col[row_start + i];
        }
    }
    
    if (idx_tau >= Ltau) return;

    __syncthreads();

    // Compute the output
    for (int64_t b = 0; b < bs; ++b) {
        scalar_t sum = make_cuFloatComplex(0.0f, 0.0f);
        for (int64_t i = 0; i < row_size; ++i) {
            auto col = s_col[tx][i];
            auto val = s_val[tx][i];
            auto shift = col / Vs - idx_tau;
            // Though if-control is entry-dependent, it only depends on a preconditioner, which is fixed.
            if (shift < -PAD) {
                shift += Ltau;  // [0, 16, 32, 48, 112, 128, 144], row_size = 7, 4x4x10
            } else if (shift > PAD) {
                shift -= Ltau; // [15, 63, 79, 95, 111, 127, 143, 159], row_size = 7, 4x4x10
            }
            sum = cuCaddf(sum, cuCmulf(val, s_input_tile[b][PAD + tx + shift])); // PAD + tx + shift \in [0, BLOCK_WIDTH-1 + 2*PAD]
        }
        out[b * stride_tau_vs + idx_tau * stride_vs + idx_site] = sum;
    }
} // precon_vec_kernel
} // namespace cuda_pcg

torch::Tensor precon_vec(
    const torch::Tensor& d_r,        // [bs, Ltau * Vs] complex64
    const torch::Tensor& precon,     // [Ltau * Vs, Ltau * Vs] complex64, sparse_csr
    int64_t Lx) 
{
    TORCH_CHECK(d_r.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(precon.is_cuda(), "Kernel must be a CUDA tensor");
    TORCH_CHECK(d_r.scalar_type() == at::ScalarType::ComplexFloat, "Input tensor must be of type ComplexFloat");

    auto precon_crow = precon.crow_indices();
    auto precon_col = precon.col_indices();
    auto precon_val = precon.values();

    auto out = torch::empty_like(d_r);
    
    auto bs = d_r.size(0);
    auto Vs = Lx * Lx;  // typically 10x10 = 100, up to 24x24 = 576
    auto Ltau = precon.size(0) / Vs;  // typically 400, up to 24x40 = 960
    
    dim3 block = {BLOCK_WIDTH};  
    auto num_blocks = ceil_div(Ltau, BLOCK_WIDTH); 
    dim3 grid = {Vs, num_blocks}; 

    // // Verify that grid/block sizes are within device limits using:
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    using scalar_t = cuFloatComplex;
    if (d_r.dtype() == at::ScalarType::ComplexFloat) {
        using scalar_t = cuFloatComplex; 
    } else if (d_r.dtype() == at::ScalarType::ComplexDouble) {
        using scalar_t = cuDoubleComplex;
    } else {
        throw std::invalid_argument("Unsupported data type");
    }

    int64_t static_shared_mem = sizeof(scalar_t) * STENCIL_SIZE * NUM_ENTRY_PER_ROW  // s_val
                          + sizeof(int64_t) * STENCIL_SIZE * NUM_ENTRY_PER_ROW; // s_col
    int64_t dynamic_shared_mem = sizeof(scalar_t) * TILE_SIZE * bs; // s_input_tile_1d
    // int64_t total_shared_mem = static_shared_mem + dynamic_shared_mem;
    // printf("Static shared memory: %d bytes, Dynamic shared memory: %d bytes, Total shared memory: %d bytes\n", 
    //        static_shared_mem, dynamic_shared_mem, total_shared_mem);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cuda_pcg::precon_vec_kernel<scalar_t><<<grid, block, dynamic_shared_mem, stream>>>(
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
    
    cudaStreamCaptureStatus capture_status;
    cudaStreamIsCapturing(stream, &capture_status);
    if (capture_status != cudaStreamCaptureStatusActive &&
    capture_status != cudaStreamCaptureStatusInvalidated) {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "CUDA stream synchronization failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA kernel execution failed");
        }
    }

    return out;
}
