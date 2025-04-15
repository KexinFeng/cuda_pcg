#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/complex.h>
#include "cuda_pcg.h"
#include "utils.h"

// #define BLOCK_SIZE 32  // 8x8, limit 1024: 32x32, 512: 24x24, 256: 16x16, 128: 10x10

#define SWAP_IN_OUT \
    tmp = interm_vec_in; \
    interm_vec_in = interm_vec_out; \
    interm_vec_out = tmp; 

namespace cuda_pcg {
template<typename scalar_t>
__global__ void b_vec_per_tau_kernel(
    const float* __restrict__ boson,      // [Ltau * Vs * 2] float32 
    const scalar_t* __restrict__ vec,     // [Vs] complex64
    scalar_t* __restrict__ out,           // [Vs] complex64
    const int64_t Lx,     
    const float dtau)
{
    extern __shared__ scalar_t smem[];  // size: [Lx, Lx] * 2
    scalar_t* interm_vec_in = smem;
    scalar_t* interm_vec_out = &smem[Lx*Lx];
    scalar_t* tmp; 

    int64_t Ltau = gridDim.x;
    int64_t bw = blockDim.x;

    int64_t stride_vs = Lx * Lx;

    int64_t tx = threadIdx.x;  
    int64_t ty = threadIdx.y;

    // Load to shared memory
    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw); offset_x++) {
            int64_t global_x = offset_x * bw + tx;
            int64_t global_y = offset_y * bw + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }
            interm_vec_in[global_y * Lx + global_x] = vec[global_y * Lx + global_x];
        }
    }
    __syncthreads();

    // boson [Ltau, Ly, Lx, 2]
    // vec [Ltau, Ly, Lx]
    // center [Lx/2, Lx/2]
    int64_t stride_lx_2 = Lx * 2;

    // // fam4
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
        // fam4: y
        int64_t idx_boson = mod(global_y - 1, Lx) * stride_lx_2 + global_x * 2 + 1;
        int64_t i_vec = mod(global_y - 1, Lx) * Lx + global_x;
        int64_t j_vec = global_y * Lx + global_x;

        mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }   
    __syncthreads();

    // // fam3
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
        // fam3: x
        int64_t idx_boson = global_y * stride_lx_2 + mod(global_x - 1, Lx) * 2 + 0;
        int64_t i_vec = global_y * Lx + mod(global_x - 1, Lx);
        int64_t j_vec = global_y * Lx + global_x;

        mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // // fam2
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
        // fam1: y
        int64_t idx_boson = global_y * stride_lx_2 + global_x * 2 + 1;
        int64_t i_vec = global_y * Lx + global_x;
        int64_t j_vec = mod(global_y + 1, Lx) * Lx + global_x;

        mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // // fam1
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
            // fam1: x
            int64_t idx_boson = global_y * stride_lx_2 + global_x * 2 + 0;
            int64_t i_vec = global_y * Lx + global_x;
            int64_t j_vec = global_y * Lx + mod(global_x + 1, Lx);

            mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau);
        }
    }
    __syncthreads();

    // fam2
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
            // fam1: y
            int64_t idx_boson = global_y * stride_lx_2 + global_x * 2 + 1;
            int64_t i_vec = global_y * Lx + global_x;
            int64_t j_vec = mod(global_y + 1, Lx) * Lx + global_x;

            mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // fam3
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
            // fam3: x
            int64_t idx_boson = global_y * stride_lx_2 + mod(global_x - 1, Lx) * 2 + 0;
            int64_t i_vec = global_y * Lx + mod(global_x - 1, Lx);
            int64_t j_vec = global_y * Lx + global_x;

            mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // fam4
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
        // fam4: y
        int64_t idx_boson = mod(global_y - 1, Lx) * stride_lx_2 + global_x * 2 + 1;
        int64_t i_vec = mod(global_y - 1, Lx) * Lx + global_x;
        int64_t j_vec = global_y * Lx + global_x;

        mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // Export to out
    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw); offset_x++) {
            int64_t global_x = offset_x * bw + tx;
            int64_t global_y = offset_y * bw + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }
            out[global_y * Lx + global_x] = interm_vec_out[global_y * Lx + global_x];
        }
    }
} // b_vec_per_tau_kernel

template<typename scalar_t>
__global__ void b_vec_per_tau_interm_out_kernel(
    const float* __restrict__ boson,      // [Ltau * Vs * 2] float32 
    const scalar_t* __restrict__ vec,     // [Vs] complex64
    scalar_t* __restrict__ out,           // [6, Vs] complex64
    const int64_t Lx,     
    const float dtau)
{
    extern __shared__ scalar_t smem[];  // size: [Lx, Lx] * 2
    scalar_t* interm_vec_in = smem;
    scalar_t* interm_vec_out = &smem[Lx*Lx];
    scalar_t* tmp; 

    int64_t Ltau = gridDim.x;
    int64_t bw = blockDim.x;

    int64_t stride_vs = Lx * Lx;

    int64_t tx = threadIdx.x;  
    int64_t ty = threadIdx.y;

    // Load to shared memory
    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw); offset_x++) {
            int64_t global_x = offset_x * bw + tx;
            int64_t global_y = offset_y * bw + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }
            interm_vec_in[global_y * Lx + global_x] = vec[global_y * Lx + global_x];
        }
    }
    __syncthreads();

    // boson [Ltau, Ly, Lx, 2]
    // vec [Ltau, Ly, Lx]
    // center [Lx/2, Lx/2]
    int64_t stride_lx_2 = Lx * 2;

    // // fam4
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
        // fam4: y
        int64_t idx_boson = mod(global_y - 1, Lx) * stride_lx_2 + global_x * 2 + 1;
        int64_t i_vec = mod(global_y - 1, Lx) * Lx + global_x;
        int64_t j_vec = global_y * Lx + global_x;

        mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }   
    __syncthreads();

    // Export to out
    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw); offset_x++) {
            int64_t global_x = offset_x * bw + tx;
            int64_t global_y = offset_y * bw + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }
            out[0 * stride_vs + global_y * Lx + global_x] = interm_vec_out[global_y * Lx + global_x];
        }
    }

    // // fam3
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
        // fam3: x
        int64_t idx_boson = global_y * stride_lx_2 + mod(global_x - 1, Lx) * 2 + 0;
        int64_t i_vec = global_y * Lx + mod(global_x - 1, Lx);
        int64_t j_vec = global_y * Lx + global_x;

        mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // Export to out
    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw); offset_x++) {
            int64_t global_x = offset_x * bw + tx;
            int64_t global_y = offset_y * bw + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }
            out[1 * stride_vs + global_y * Lx + global_x] = interm_vec_out[global_y * Lx + global_x];
        }
    }

    // // fam2
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
        // fam1: y
        int64_t idx_boson = global_y * stride_lx_2 + global_x * 2 + 1;
        int64_t i_vec = global_y * Lx + global_x;
        int64_t j_vec = mod(global_y + 1, Lx) * Lx + global_x;

        mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // Export to out
    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw); offset_x++) {
            int64_t global_x = offset_x * bw + tx;
            int64_t global_y = offset_y * bw + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }
            out[2 * stride_vs + global_y * Lx + global_x] = interm_vec_out[global_y * Lx + global_x];
        }
    }

    // // fam1
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
            // fam1: x
            int64_t idx_boson = global_y * stride_lx_2 + global_x * 2 + 0;
            int64_t i_vec = global_y * Lx + global_x;
            int64_t j_vec = global_y * Lx + mod(global_x + 1, Lx);

            mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau);
        }
    }
    __syncthreads();

    // Export to out
    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw); offset_x++) {
            int64_t global_x = offset_x * bw + tx;
            int64_t global_y = offset_y * bw + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }
            out[3 * stride_vs + global_y * Lx + global_x] = interm_vec_out[global_y * Lx + global_x];
        }
    }

    // fam2
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
            // fam1: y
            int64_t idx_boson = global_y * stride_lx_2 + global_x * 2 + 1;
            int64_t i_vec = global_y * Lx + global_x;
            int64_t j_vec = mod(global_y + 1, Lx) * Lx + global_x;

            mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // Export to out
    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw); offset_x++) {
            int64_t global_x = offset_x * bw + tx;
            int64_t global_y = offset_y * bw + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }
            out[4 * stride_vs + global_y * Lx + global_x] = interm_vec_out[global_y * Lx + global_x];
        }
    }

    // fam3
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
            // fam3: x
            int64_t idx_boson = global_y * stride_lx_2 + mod(global_x - 1, Lx) * 2 + 0;
            int64_t i_vec = global_y * Lx + mod(global_x - 1, Lx);
            int64_t j_vec = global_y * Lx + global_x;

            mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // Export to out
    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw); offset_x++) {
            int64_t global_x = offset_x * bw + tx;
            int64_t global_y = offset_y * bw + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }
            out[5 * stride_vs + global_y * Lx + global_x] = interm_vec_out[global_y * Lx + global_x];
        }
    }
} // b_vec_per_tau_iterm_out_kernel
} // namespace cuda_pcg

torch::Tensor b_vec_per_tau(
    const torch::Tensor& boson,   // [Vs * 2] float32
    const torch::Tensor& vec,     // [Vs] complex64
    const int64_t Lx,     
    const float dtau,
    const bool interm_out_bool = false)
{
    TORCH_CHECK(boson.dim() == 1, "Boson tensor must have 1 dimension: [Ltau * Vs * 2]");
    TORCH_CHECK(vec.dim() == 1, "Input tensor must have 1 dimension: [Vs]");
    TORCH_CHECK(boson.size(0) == vec.size(0) * 2, "Boson tensor's size must be twice the size of vec's size");

    TORCH_CHECK(vec.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(boson.is_cuda(), "Boson must  CUDA tensor");
    TORCH_CHECK(vec.scalar_type() == at::ScalarType::ComplexFloat, "Input tensor must be of type ComplexFloat");
    TORCH_CHECK(boson.scalar_type() == at::ScalarType::Float, "Boson tensor must be of type Float");

    torch::Tensor vec_in = vec;
    torch::Tensor out;

    int64_t Vs = Lx * Lx;

    using scalar_t = cuFloatComplex;
    if (vec.dtype() == at::ScalarType::ComplexFloat) {
        using scalar_t = cuFloatComplex; 
    } else if (vec.dtype() == at::ScalarType::ComplexDouble) {
        using scalar_t = cuDoubleComplex;
    } else {
        throw std::invalid_argument("Unsupported data type");
    }

    cudaError_t kernel_err;
    cudaError_t err;

    // B_vec_mul
    dim3 block = {4, ceil_div(BLOCK_SIZE, 4)};
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        
    if (!interm_out_bool) {
        out = torch::empty_like(vec);

        cuda_pcg::b_vec_per_tau_kernel<<<1, block, 2 * Vs * sizeof(scalar_t), stream>>>(
            reinterpret_cast<float*>(boson.data_ptr()),
            reinterpret_cast<scalar_t*>(vec_in.data_ptr()),
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            Lx, dtau);

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(kernel_err) << std::endl;
            throw std::runtime_error("CUDA kernel launch failed");
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "CUDA stream synchronization failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA kernel execution failed");
        }

    } else {
        out = torch::empty_like(vec);     
        out = out.repeat({6});

        cuda_pcg::b_vec_per_tau_interm_out_kernel<<<1, block, 2 * Vs * sizeof(scalar_t), stream>>>(
            reinterpret_cast<float*>(boson.data_ptr()),
            reinterpret_cast<scalar_t*>(vec_in.data_ptr()),
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            Lx, dtau);

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(kernel_err) << std::endl;
            throw std::runtime_error("CUDA kernel launch failed");
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "CUDA stream synchronization failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA kernel execution failed");
        }
    }
    
    return out;
}