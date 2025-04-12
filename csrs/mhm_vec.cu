#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/complex.h>
#include "cuda_pcg.h"
#include "utils.h"


#define BLOCK_WIDTH 8  // 8x8, limit 1024: 32x32, 512: 24x24, 256: 16x16, 128: 10x10

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
    int64_t bs = gridDim.y;
    int64_t bw = blockDim.x;

    int64_t stride_vs = Lx * Lx;
    int64_t stride_tau_vs = stride_vs * Ltau;

    int64_t tx = threadIdx.x;  
    int64_t ty = threadIdx.y;
    int64_t tau = blockIdx.x;
    int64_t b = blockIdx.y;

    // Load to shared memory
    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw); offset_x++) {
            int64_t global_x = offset_x * bw + tx;
            int64_t global_y = offset_y * bw + ty;
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
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;

            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }

            // fam4: y
            int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + mod(global_y - 1, Lx) * stride_lx_2 + global_x * 2 + 1;
            int64_t i_vec = mod(global_y - 1, Lx) * Lx + global_x;
            int64_t j_vec = global_y * Lx + global_x;

            float boson_val = boson[idx_boson];
            cuFloatComplex cosh_dtau = make_cuFloatComplex(coshf(dtau/2), 0.0f);
            cuFloatComplex sinh_dtau = make_cuFloatComplex(sinhf(dtau/2), 0.0f);
            float cos_boson = cosf(boson_val);
            float sin_boson = sinf(boson_val);
            cuFloatComplex sinh_exp_pos = sinh_dtau * make_cuFloatComplex(cos_boson, sin_boson);  // exp(1i * boson_val)
            cuFloatComplex sinh_exp_neg = sinh_dtau * make_cuFloatComplex(cos_boson, -sin_boson);  // exp(-1i * boson_val)
            if (i_vec < stride_vs && j_vec < stride_vs) {
                interm_vec_out[i_vec] = cosh_dtau * interm_vec_in[i_vec] + sinh_exp_pos * interm_vec_in[j_vec];
                interm_vec_out[j_vec] = cosh_dtau * interm_vec_in[j_vec] + sinh_exp_neg * interm_vec_in[i_vec];
            }
        }
    }
    __syncthreads();

    // // fam3
    tmp = interm_vec_in;
    interm_vec_in = interm_vec_out;
    interm_vec_out = tmp;
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;

            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }

            // fam3: x
            int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + global_y * stride_lx_2 + mod(global_x - 1, Lx) * 2 + 0;
            int64_t i_vec = global_y * Lx + mod(global_x - 1, Lx);
            int64_t j_vec = global_y * Lx + global_x;

            float boson_val = boson[idx_boson];
            cuFloatComplex cosh_dtau = make_cuFloatComplex(coshf(dtau/2), 0.0f);
            cuFloatComplex sinh_dtau = make_cuFloatComplex(sinhf(dtau/2), 0.0f);
            float cos_boson = cosf(boson_val);
            float sin_boson = sinf(boson_val);
            cuFloatComplex sinh_exp_pos = sinh_dtau * make_cuFloatComplex(cos_boson, sin_boson);  // exp(1i * boson_val)
            cuFloatComplex sinh_exp_neg = sinh_dtau * make_cuFloatComplex(cos_boson, -sin_boson);  // exp(-1i * boson_val)
            if (i_vec < stride_vs && j_vec < stride_vs) {
                interm_vec_out[i_vec] = cosh_dtau * interm_vec_in[i_vec] + sinh_exp_pos * interm_vec_in[j_vec];
                interm_vec_out[j_vec] = cosh_dtau * interm_vec_in[j_vec] + sinh_exp_neg * interm_vec_in[i_vec];
            }
        }
    }
    __syncthreads();

    // fam2
    tmp = interm_vec_in;
    interm_vec_in = interm_vec_out;
    interm_vec_out = tmp;
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;

            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }

            // fam1: y
            int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + global_y * stride_lx_2 + global_x * 2 + 1;
            int64_t i_vec = global_y * Lx + global_x;
            int64_t j_vec = mod(global_y + 1, Lx) * Lx + global_x;

            float boson_val = boson[idx_boson];
            cuFloatComplex cosh_dtau = make_cuFloatComplex(coshf(dtau/2), 0.0f);
            cuFloatComplex sinh_dtau = make_cuFloatComplex(sinhf(dtau/2), 0.0f);
            float cos_boson = cosf(boson_val);
            float sin_boson = sinf(boson_val);
            cuFloatComplex sinh_exp_pos = sinh_dtau * make_cuFloatComplex(cos_boson, sin_boson);  // exp(1i * boson_val)
            cuFloatComplex sinh_exp_neg = sinh_dtau * make_cuFloatComplex(cos_boson, -sin_boson);  // exp(-1i * boson_val)
            if (i_vec < stride_vs && j_vec < stride_vs) {
                interm_vec_out[i_vec] = cosh_dtau * interm_vec_in[i_vec] + sinh_exp_pos * interm_vec_in[j_vec];
                interm_vec_out[j_vec] = cosh_dtau * interm_vec_in[j_vec] + sinh_exp_neg * interm_vec_in[i_vec];
            }
        }
    }
    __syncthreads();

    // // fam1
    tmp = interm_vec_in;
    interm_vec_in = interm_vec_out;
    interm_vec_out = tmp;
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx/2, bw); cntr_offset_x++) {
            // Slide the block over the family centers of a rectangle shape [Lx/2, Lx]
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;

            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }         

            // fam1: x
            int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + global_y * stride_lx_2 + global_x * 2 + 0;
            int64_t i_vec = global_y * Lx + global_x;
            int64_t j_vec = global_y * Lx + mod(global_x + 1, Lx);

            // interm_vec_out[i_vec] = cosh(dtau) * interm_vec_in[i_vec] + sinh(dtau) * exp(1i * boson[idx_boson]) * interm_vec_in[j_vec];
            // interm_vec_out[j_vec] = cosh(dtau) * interm_vec_in[j_vec] + sinh(dtau) * exp(-1i * boson[idx_boson]) * interm_vec_in[i_vec];                    
            float boson_val = boson[idx_boson];
            cuFloatComplex cosh_dtau = make_cuFloatComplex(coshf(dtau), 0.0f);
            cuFloatComplex sinh_dtau = make_cuFloatComplex(sinhf(dtau), 0.0f);
            float cos_boson = cosf(boson_val);
            float sin_boson = sinf(boson_val);
            cuFloatComplex sinh_exp_pos = sinh_dtau * make_cuFloatComplex(cos_boson, sin_boson);  // exp(1i * boson_val)
            cuFloatComplex sinh_exp_neg = sinh_dtau * make_cuFloatComplex(cos_boson, -sin_boson);  // exp(-1i * boson_val)
            if (i_vec < stride_vs && j_vec < stride_vs) {
                interm_vec_out[i_vec] = cosh_dtau * interm_vec_in[i_vec] + sinh_exp_pos * interm_vec_in[j_vec];
                interm_vec_out[j_vec] = cosh_dtau * interm_vec_in[j_vec] + sinh_exp_neg * interm_vec_in[i_vec];
            }
        }
    }
    __syncthreads();

    // fam2
    tmp = interm_vec_in;
    interm_vec_in = interm_vec_out;
    interm_vec_out = tmp;
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;

            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }

            // fam1: y
            int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + global_y * stride_lx_2 + global_x * 2 + 1;
            int64_t i_vec = global_y * Lx + global_x;
            int64_t j_vec = mod(global_y + 1, Lx) * Lx + global_x;

            float boson_val = boson[idx_boson];
            cuFloatComplex cosh_dtau = make_cuFloatComplex(coshf(dtau/2), 0.0f);
            cuFloatComplex sinh_dtau = make_cuFloatComplex(sinhf(dtau/2), 0.0f);
            float cos_boson = cosf(boson_val);
            float sin_boson = sinf(boson_val);
            cuFloatComplex sinh_exp_pos = sinh_dtau * make_cuFloatComplex(cos_boson, sin_boson);  // exp(1i * boson_val)
            cuFloatComplex sinh_exp_neg = sinh_dtau * make_cuFloatComplex(cos_boson, -sin_boson);  // exp(-1i * boson_val)
            if (i_vec < stride_vs && j_vec < stride_vs) {
                interm_vec_out[i_vec] = cosh_dtau * interm_vec_in[i_vec] + sinh_exp_pos * interm_vec_in[j_vec];
                interm_vec_out[j_vec] = cosh_dtau * interm_vec_in[j_vec] + sinh_exp_neg * interm_vec_in[i_vec];
            }
        }
    }
    __syncthreads();

    // fam3
    tmp = interm_vec_in;
    interm_vec_in = interm_vec_out;
    interm_vec_out = tmp;
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;

            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }

            // fam3: x
            int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + global_y * stride_lx_2 + mod(global_x - 1, Lx) * 2 + 0;
            int64_t i_vec = global_y * Lx + mod(global_x - 1, Lx);
            int64_t j_vec = global_y * Lx + global_x;

            float boson_val = boson[idx_boson];
            cuFloatComplex cosh_dtau = make_cuFloatComplex(coshf(dtau/2), 0.0f);
            cuFloatComplex sinh_dtau = make_cuFloatComplex(sinhf(dtau/2), 0.0f);
            float cos_boson = cosf(boson_val);
            float sin_boson = sinf(boson_val);
            cuFloatComplex sinh_exp_pos = sinh_dtau * make_cuFloatComplex(cos_boson, sin_boson);  // exp(1i * boson_val)
            cuFloatComplex sinh_exp_neg = sinh_dtau * make_cuFloatComplex(cos_boson, -sin_boson);  // exp(-1i * boson_val)
            if (i_vec < stride_vs && j_vec < stride_vs) {
                interm_vec_out[i_vec] = cosh_dtau * interm_vec_in[i_vec] + sinh_exp_pos * interm_vec_in[j_vec];
                interm_vec_out[j_vec] = cosh_dtau * interm_vec_in[j_vec] + sinh_exp_neg * interm_vec_in[i_vec];
            }
        }
    }
    __syncthreads();

    // fam4
    tmp = interm_vec_in;
    interm_vec_in = interm_vec_out;
    interm_vec_out = tmp;
    for (int64_t cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx, bw); cntr_offset_y++) {
        for (int64_t cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx / 2, bw); cntr_offset_x++) {
            int64_t cntr_x = cntr_offset_x * bw + tx;
            int64_t cntr_y = cntr_offset_y * bw + ty;

            int64_t global_y = cntr_y;
            int64_t global_x = cntr_x * 2 + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;
            }

            // fam4: y
            int64_t idx_boson = b * stride_tau_vs_2 + tau * stride_vs_2 + mod(global_y - 1, Lx) * stride_lx_2 + global_x * 2 + 1;
            int64_t i_vec = mod(global_y - 1, Lx) * Lx + global_x;
            int64_t j_vec = global_y * Lx + global_x;

            float boson_val = boson[idx_boson];
            cuFloatComplex cosh_dtau = make_cuFloatComplex(coshf(dtau/2), 0.0f);
            cuFloatComplex sinh_dtau = make_cuFloatComplex(sinhf(dtau/2), 0.0f);
            float cos_boson = cosf(boson_val);
            float sin_boson = sinf(boson_val);
            cuFloatComplex sinh_exp_pos = sinh_dtau * make_cuFloatComplex(cos_boson, sin_boson);  // exp(1i * boson_val)
            cuFloatComplex sinh_exp_neg = sinh_dtau * make_cuFloatComplex(cos_boson, -sin_boson);  // exp(-1i * boson_val)
            if (i_vec < stride_vs && j_vec < stride_vs) {
                interm_vec_out[i_vec] = cosh_dtau * interm_vec_in[i_vec] + sinh_exp_pos * interm_vec_in[j_vec];
                interm_vec_out[j_vec] = cosh_dtau * interm_vec_in[j_vec] + sinh_exp_neg * interm_vec_in[i_vec];
            }
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
    int64_t bw = blockDim.x;

    int64_t stride_vs = Lx * Lx;
    int64_t stride_tau_vs = stride_vs * Ltau;

    int64_t tx = threadIdx.x;  
    int64_t ty = threadIdx.y;
    int64_t tau = blockIdx.x;
    int64_t b = blockIdx.y;

    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw); offset_x++) {
            int64_t global_x = offset_x * bw + tx;
            int64_t global_y = offset_y * bw + ty;
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
    int64_t bw = blockDim.x;

    int64_t stride_vs = Lx * Lx;
    int64_t stride_tau_vs = stride_vs * Ltau;

    int64_t tx = threadIdx.x;  
    int64_t ty = threadIdx.y;
    int64_t tau = blockIdx.x;
    int64_t b = blockIdx.y;

    for (int64_t offset_y = 0; offset_y < ceil_div(Lx, bw); offset_y++) {
        for (int64_t offset_x = 0; offset_x < ceil_div(Lx, bw); offset_x++) {
            int64_t global_x = offset_x * bw + tx;
            int64_t global_y = offset_y * bw + ty;
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

torch::Tensor mhm_vec(
    const torch::Tensor& boson,   // [bs, Ltau * Vs * 2] float32
    const torch::Tensor& vec,     // [bs, Ltau * Vs] complex64
    const int64_t Lx,
    const float dtau)
{
    TORCH_CHECK(boson.dim() == 2, "Boson tensor must have 2 dimensions: [bs, Ltau * Vs * 2]");
    TORCH_CHECK(vec.dim() == 2, "Input tensor must have 2 dimensions: [bs, Ltau * Vs]");
    TORCH_CHECK(boson.size(0) == vec.size(0), "Batch size of boson and vec tensors must match");
    TORCH_CHECK(boson.size(1) == vec.size(1) * 2, "Boson tensor's second dimension must be twice the size of vec's second dimension");

    TORCH_CHECK(vec.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(boson.is_cuda(), "Boson must  CUDA tensor");
    TORCH_CHECK(vec.scalar_type() == at::ScalarType::ComplexFloat, "Input tensor must be of type ComplexFloat");
    TORCH_CHECK(boson.scalar_type() == at::ScalarType::Float, "Boson tensor must be of type Float");
    TORCH_CHECK(boson.is_contiguous(), "Boson tensor must be contiguous");

    auto vec_in = vec;
    auto out1 = torch::empty_like(vec);
    auto out2 = torch::empty_like(vec);
    auto interm_vec = torch::empty_like(vec);

    auto bs = vec.size(0);
    auto Vs = Lx * Lx;
    auto Ltau = vec.size(1) / Vs; 

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
    dim3 block = {BLOCK_WIDTH, BLOCK_WIDTH};
    dim3 grid = {Ltau, bs};
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cuda_pcg::mhm_vec_kernel<<<grid, block, 2 * Vs * sizeof(scalar_t), stream>>>(
        reinterpret_cast<float*>(boson.data_ptr()),
        reinterpret_cast<scalar_t*>(vec_in.data_ptr()),
        reinterpret_cast<scalar_t*>(out1.data_ptr()),
        Lx, dtau, 0);
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

    // vec_minus_B_vec
    cuda_pcg::vec_minus_B_vec_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<scalar_t*>(vec_in.data_ptr()),
        reinterpret_cast<scalar_t*>(out1.data_ptr()),
        reinterpret_cast<scalar_t*>(out2.data_ptr()),
        Lx);
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

    vec_in = out2;
    cudaMemcpyAsync(
        interm_vec.data_ptr(),
        vec_in.data_ptr(),
        vec_in.numel() * sizeof(scalar_t),
        cudaMemcpyDeviceToDevice,
        stream);

    // B_vec_mul
    cuda_pcg::mhm_vec_kernel<<<grid, block, 2 * Vs * sizeof(scalar_t), stream>>>(
        reinterpret_cast<float*>(boson.data_ptr()),
        reinterpret_cast<scalar_t*>(vec_in.data_ptr()),
        reinterpret_cast<scalar_t*>(out1.data_ptr()),
        Lx, dtau, 1);
        
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

    // vec_minus_B_vec
    cuda_pcg::vec_minus_B_vec_2_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<scalar_t*>(vec_in.data_ptr()),
        reinterpret_cast<scalar_t*>(out1.data_ptr()),
        reinterpret_cast<scalar_t*>(out2.data_ptr()),
        Lx);

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

    out2 = torch::cat({out2, interm_vec}, 0);
    return out2;      
}