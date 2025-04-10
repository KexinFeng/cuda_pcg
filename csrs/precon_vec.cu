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
#define BLOCK_WIDTH 32
#define PAD 6
#define TILE_SIZE (BLOCK_WIDTH + 2 * PAD)
#define STENCIL_SIZE BLOCK_WIDTH
#define NUM_ENTRY_PER_ROW 13 // PAD * 2 + 1
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
    __shared__ scalar_t s_val[STENCIL_SIZE][NUM_ENTRY_PER_ROW];
    __shared__ int64_t s_col[STENCIL_SIZE][NUM_ENTRY_PER_ROW];
    __shared__ scalar_t s_input_tile[TILE_SIZE];

    int tx = threadIdx.x; 
    int idx_tau = blockIdx.y * blockDim.x + tx;  // global temporal idx: [blockIdx.y, threadIdx.x]
    // BlockIdx.y ranges (num_blocks);
    // BlockDim.x == BLOCK_WIDTH, i.e. block_size_x

    int idx_site = blockIdx.x;
    int stride_vs = Lx * Lx;
    if (idx_site >= stride_vs) return;

    int Vs = Lx * Lx; 

    // Load the input into shared memory
    int max_tx_num = std::min(Ltau - blockIdx.y * blockDim.x, blockDim.x);
    if (tx < max_tx_num) {
        s_input_tile[PAD + tx] = d_r[idx_tau * stride_vs + idx_site]; 
    }
  
    if (tx < PAD) { // Left halo
        // Possible that max_tx_num < PAD, but it's ok
        int idx_tau_pad = mod(idx_tau - PAD, Ltau);  // left shift each idx_tau by PAD
        s_input_tile[tx] = d_r[idx_tau_pad * stride_vs + idx_site]; 
    }
    
    // if (tx >= blockDim.x - PAD) { // Right halo; backward count by PAD
    //     int idx_tau_pad = mod(idx_tau + PAD, Ltau);  // right shift each idx_tau by PAD
    //     s_input_tile[tx + 2*PAD] = /* starting at blockDim.x + PAD when tx = blockDim.x - PAD */
    //     d_r[idx_tau_pad * stride_vs + idx_site];
    // }
    if (tx >= blockDim.x - PAD) { // Right halo
        // Delta = tx - (blockDim.x - PAD)
        // idx_tau_pad = mod(blockIdx.y * blockDim.x + max_tx_num + Delta, Ltau)
        // d_r[idx_tau_pad * stride_vs + idx_site]
        // s_input_tile[PAD + max_tx_num + Delta]
        int idx_tau_pad = mod(blockIdx.y * blockDim.x + max_tx_num + (tx - (blockDim.x - PAD)), Ltau); 
        s_input_tile[PAD + max_tx_num + (tx - (blockDim.x - PAD))] = d_r[idx_tau_pad * stride_vs + idx_site];
    }

    if (idx_tau >= Ltau) return;

    // Load stencil into shared memory
    int row_start = precon_crow[idx_tau * stride_vs + idx_site];
    int row_size = precon_crow[idx_tau * stride_vs + idx_site + 1] - row_start;  // ~ Num_ENTRY_PER_ROW
    for (int i = 0; i < row_size; ++i) {
        s_val[tx][i] = precon_val[row_start + i];  // [Ltau * Vs], tx->row of shared mat, i->col of shared mat
        s_col[tx][i] = precon_col[row_start + i];
    }

    __syncthreads();

    if (blockIdx.y == gridDim.y - 1 && tx == 10 - 1 && blockIdx.x == 0){
        printf("==>Debug: Code reached here. \nBlockIdx.y: %d, ThreadIdx.x: %d\n", blockIdx.y, tx);
        printf("BlockIdx.x: %d\n", blockIdx.x);
        printf("s_col[0]: ");
        
        printf("\ns_input_tile: ");
        for (int i = 0; i < TILE_SIZE; ++i) {
            printf("(%f, %f)\n", cuCrealf(s_input_tile[i]), cuCimagf(s_input_tile[i]));
        }
        printf("\n");
    }

    // Compute the output
    scalar_t sum = make_cuFloatComplex(0.0f, 0.0f);
    for (int i = 0; i < row_size; ++i) {
        auto col = s_col[tx][i];
        auto val = s_val[tx][i];
        auto shift = col / Vs - idx_tau;
        // Though if-control is entry-dependent, it only depends on a preconditioner, which is fixed.
        if (shift < -PAD) {
            shift += Ltau;  // [0, 16, 32, 48, 112, 128, 144], row_size = 7, 4x4x10
        } else if (shift > PAD) {
            shift -= Ltau; // [15, 63, 79, 95, 111, 127, 143, 159], row_size = 7, 4x4x10
        }
        sum = cuCaddf(sum, cuCmulf(val, s_input_tile[PAD + tx + shift])); // PAD + tx + shift \in [0, BLOCK_WIDTH-1 + 2*PAD]
        
        if (blockIdx.y == gridDim.y - 1 && tx == 10 - 1 && blockIdx.x == 0) {
            printf("--------> \nidx_tau: %d, idx_site: %d, s_val[%d][%d]: %f + %fi, s_input_tile[%d]: %f + %fi\nshift: %d col: %d\n", 
                idx_tau, idx_site, tx, i, 
                cuCrealf(s_val[tx][i]), cuCimagf(s_val[tx][i]), 
                PAD + tx + shift,
                cuCrealf(s_input_tile[PAD + tx + shift]), cuCimagf(s_input_tile[PAD + tx + shift]), 
                shift, col);
        }
    }
    out[idx_tau * stride_vs + idx_site] = sum;

    // // Compute the output
    // scalar_t sum = make_cuFloatComplex(0.0f, 0.0f);
    // int half_row_size = row_size / 2;
    // int left_roll = idx_tau - half_row_size; 
    // int right_roll = idx_tau + half_row_size;
    // for (int i = 0; i < row_size && i < NUM_ENTRY_PER_ROW; ++i) {       
    //     int j = i;
    //     // These if-controls are structural and entry-value-independent, thus can be captured by cuda_graph.
    //     if (left_roll < 0) 
    //         // First row of precon may be wrapped around
    //         // Columns: [0, 16, 32, 48, 112, 128, 144], row_size = 7, 4x4x10
    //         j = mod(i + left_roll, row_size); 
    //     if (right_roll >= Ltau) 
    //         // Last row: [15, 63, 79, 95, 111, 127, 143, 159], row_size = 7, 4x4x10
    //         j = mod(i + (right_roll - Ltau + 1), row_size);

    //     int shift = i - half_row_size;  // [-PAD (or half_row_size), ..., PAD (or half_row_size)]
    //     sum = cuCaddf(sum, cuCmulf(s_val[tx][j], s_input_tile[PAD + tx + shift])); // PAD + tx + shift \in [0, BLOCK_WIDTH + 2*PAD - 1]
        
    //     if (blockIdx.y == 0 && tx == 0 && blockIdx.x == 0) {
    //         printf("idx_tau: %d, idx_site: %d, s_val[%d][%d]: %f + %fi, s_input_tile[%d]: %f + %fi\nshift: %d\n", 
    //             idx_tau, idx_site, tx, i, 
    //             cuCrealf(s_val[tx][j]), cuCimagf(s_val[tx][j]), 
    //             PAD + tx + shift,
    //             cuCrealf(s_input_tile[PAD + tx + shift]), cuCimagf(s_input_tile[PAD + tx + shift]), 
    //             shift);
    //     }

    //     if (blockIdx.y == 1 && tx == blockDim.x - 1 && blockIdx.x == 0) {
    //         printf("idx_tau: %d, idx_site: %d, s_val[%d][%d]: %f + %fi, s_input_tile[%d]: %f + %fi\nshift: %d\n", 
    //             idx_tau, idx_site, tx, i, 
    //             cuCrealf(s_val[tx][j]), cuCimagf(s_val[tx][j]), 
    //             PAD + tx + shift,
    //             cuCrealf(s_input_tile[PAD + tx + shift]), cuCimagf(s_input_tile[PAD + tx + shift]), 
    //             shift);
    //     }
    // }
    // out[idx_tau * stride_vs + idx_site] = sum;
} // precon_vec_kernel
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

    int shared_mem = (sizeof(scalar_t) * STENCIL_SIZE * NUM_ENTRY_PER_ROW  // s_val
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
