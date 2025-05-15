#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/complex.h>
#include "utils.h"

#define SWAP_IN_OUT \
    tmp = interm_vec_in; \
    interm_vec_in = interm_vec_out; \
    interm_vec_out = tmp; 
    
namespace cuda_pcg {
template<typename scalar_t>
__global__ void mhm_vec_kernel(
    const float* __restrict__ boson,      // [bs, Ltau * Vs * 2] float32 
    const scalar_t* __restrict__ vec,     // [bs, Ltau * Vs] complex64
    scalar_t* __restrict__ out,           // [bs, Ltau * Vs] complex64
    const int64_t Lx,  // typically Lx^2 = 10x10 = 100, up to 24x24 = 576
    const float dtau, 
    const int64_t tau_roll)
{
    extern __shared__ scalar_t smem[];  // size: [Lx, Lx] * 2
    scalar_t* interm_vec_in = smem;
    scalar_t* interm_vec_out = &smem[Lx*Lx];
    scalar_t* tmp; 

    int64_t Ltau = gridDim.x;
    int64_t bw_x = blockDim.x;
    int64_t bw_y = blockDim.y;

    int64_t stride_vs = Lx * Lx;
    int64_t stride_tau_vs = stride_vs * Ltau;

    int64_t tx = threadIdx.x;  
    int64_t ty = threadIdx.y;
    int64_t tau = blockIdx.x;
    int64_t b = blockIdx.y;

    // Load to shared memory
    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw_y); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw_x); offset_x++) {
            int64_t global_x = offset_x * bw_x + tx;
            int64_t global_y = offset_y * bw_y + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }
            interm_vec_in[global_y * Lx + global_x] = vec[b * stride_tau_vs + mod(tau + tau_roll, Ltau) * stride_vs + global_y * Lx + global_x];
        }
    }
    __syncthreads();

    // boson [Ltau, Ly, Lx, 2]
    // vec [Ltau, Ly, Lx]
    // center [Lx/2, Lx/2]
    int64_t stride_tau_vs_2 = Ltau * Lx * Lx * 2;
    int64_t stride_vs_2 = Lx * Lx * 2;
    int64_t stride_lx_2 = Lx * 2;

    // // fam4
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw_y); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw_x); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw_x + tx;
            int64_t cntr_y = cntr_offset_y * bw_y + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
        // fam4: y
        int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + mod(global_y - 1, Lx) * stride_lx_2 + global_x * 2 + 1;
        int64_t i_vec = mod(global_y - 1, Lx) * Lx + global_x;
        int64_t j_vec = global_y * Lx + global_x;

        mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }   
    __syncthreads();

    // // fam3
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw_y); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw_x); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw_x + tx;
            int64_t cntr_y = cntr_offset_y * bw_y + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
        // fam3: x
        int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + global_y * stride_lx_2 + mod(global_x - 1, Lx) * 2 + 0;
        int64_t i_vec = global_y * Lx + mod(global_x - 1, Lx);
        int64_t j_vec = global_y * Lx + global_x;

        mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // fam2
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw_y); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw_x); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw_x + tx;
            int64_t cntr_y = cntr_offset_y * bw_y + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
        // fam1: y
        int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + global_y * stride_lx_2 + global_x * 2 + 1;
        int64_t i_vec = global_y * Lx + global_x;
        int64_t j_vec = mod(global_y + 1, Lx) * Lx + global_x;

        mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // // fam1
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw_y); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw_x); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw_x + tx;
            int64_t cntr_y = cntr_offset_y * bw_y + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
            // fam1: x
            int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + global_y * stride_lx_2 + global_x * 2 + 0;
            int64_t i_vec = global_y * Lx + global_x;
            int64_t j_vec = global_y * Lx + mod(global_x + 1, Lx);

            mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau);
        }
    }
    __syncthreads();

    // fam2
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw_y); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw_x); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw_x + tx;
            int64_t cntr_y = cntr_offset_y * bw_y + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
            // fam1: y
            int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + global_y * stride_lx_2 + global_x * 2 + 1;
            int64_t i_vec = global_y * Lx + global_x;
            int64_t j_vec = mod(global_y + 1, Lx) * Lx + global_x;

            mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // fam3
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw_y); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw_x); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw_x + tx;
            int64_t cntr_y = cntr_offset_y * bw_y + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
            // fam3: x
            int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + global_y * stride_lx_2 + mod(global_x - 1, Lx) * 2 + 0;
            int64_t i_vec = global_y * Lx + mod(global_x - 1, Lx);
            int64_t j_vec = global_y * Lx + global_x;

            mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // fam4
    SWAP_IN_OUT
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw_y); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw_x); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw_x + tx;
            int64_t cntr_y = cntr_offset_y * bw_y + ty;
            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }
        // fam4: y
        int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + mod(global_y - 1, Lx) * stride_lx_2 + global_x * 2 + 1;
        int64_t i_vec = mod(global_y - 1, Lx) * Lx + global_x;
        int64_t j_vec = global_y * Lx + global_x;

        mat_vec_mul_2b2(boson, interm_vec_in, interm_vec_out, idx_boson, i_vec, j_vec, dtau / 2);
        }
    }
    __syncthreads();

    // Export to out
    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw_y); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw_x); offset_x++) {
            int64_t global_x = offset_x * bw_x + tx;
            int64_t global_y = offset_y * bw_y + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }
            out[b * stride_tau_vs + mod(tau + tau_roll, Ltau) * stride_vs + global_y * Lx + global_x] = interm_vec_out[global_y * Lx + global_x];
        }
    }
} // mhm_vec_kernel

template<typename scalar_t>
__global__ void vec_minus_B_vec_kernel(
    const scalar_t* __restrict__ vec,     // [bs, Ltau * Vs] complex64
    const scalar_t* __restrict__ B_vec,     // [bs, Ltau * Vs] complex64
    scalar_t* __restrict__ out,           // [bs, Ltau * Vs] complex64
    const int64_t Lx)
{
    int64_t Ltau = gridDim.x;
    int64_t bw_x = blockDim.x;
    int64_t bw_y = blockDim.y;

    int64_t stride_vs = Lx * Lx;
    int64_t stride_tau_vs = stride_vs * Ltau;

    int64_t tx = threadIdx.x;  
    int64_t ty = threadIdx.y;
    int64_t tau = blockIdx.x;
    int64_t b = blockIdx.y;

    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw_y); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw_x); offset_x++) {
            int64_t global_x = offset_x * bw_x + tx;
            int64_t global_y = offset_y * bw_y + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }

            if (tau > 0) {
                scalar_t last_tau_B_vec = B_vec[b * stride_tau_vs + (tau - 1) * stride_vs + global_y * Lx + global_x];
                out[b * stride_tau_vs + tau * stride_vs + global_y * Lx + global_x] = vec[b * stride_tau_vs + tau * stride_vs + global_y * Lx + global_x] - last_tau_B_vec;
            } else { // tau == 0
                scalar_t last_tau_B_vec = B_vec[b * stride_tau_vs + (Ltau - 1) * stride_vs + global_y * Lx + global_x];
                out[b * stride_tau_vs + tau * stride_vs + global_y * Lx + global_x] = vec[b * stride_tau_vs + tau * stride_vs + global_y * Lx + global_x] + last_tau_B_vec;
            }
        }
    }
} // vec_minus_B_vec_kernel

template<typename scalar_t>
__global__ void vec_minus_B_vec_2_kernel(
    const scalar_t* __restrict__ vec,     // [bs, Ltau * Vs] complex64
    const scalar_t* __restrict__ B_vec,     // [bs, Ltau * Vs] complex64
    scalar_t* __restrict__ out,           // [bs, Ltau * Vs] complex64
    const int64_t Lx)
{
    int64_t Ltau = gridDim.x;
    int64_t bw_x = blockDim.x;
    int64_t bw_y = blockDim.y;

    int64_t stride_vs = Lx * Lx;
    int64_t stride_tau_vs = stride_vs * Ltau;

    int64_t tx = threadIdx.x;  
    int64_t ty = threadIdx.y;
    int64_t tau = blockIdx.x;
    int64_t b = blockIdx.y;

    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw_y); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw_x); offset_x++) {
            int64_t global_x = offset_x * bw_x + tx;
            int64_t global_y = offset_y * bw_y + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }

            if (tau < Ltau - 1) {
                scalar_t last_tau_B_vec = B_vec[b * stride_tau_vs + (tau + 1) * stride_vs + global_y * Lx + global_x];
                out[b * stride_tau_vs + tau * stride_vs + global_y * Lx + global_x] = vec[b * stride_tau_vs + tau * stride_vs + global_y * Lx + global_x] - last_tau_B_vec;
            } else { // tau == Ltau - 1
                scalar_t last_tau_B_vec = B_vec[b * stride_tau_vs + 0 * stride_vs + global_y * Lx + global_x];
                out[b * stride_tau_vs + tau * stride_vs + global_y * Lx + global_x] = vec[b * stride_tau_vs + tau * stride_vs + global_y * Lx + global_x] + last_tau_B_vec;
            }
        }
    }
} // vec_minus_B_vec_2_kernel

} // namespace cuda_pcg