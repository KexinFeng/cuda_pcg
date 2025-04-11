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
    const scalar_t* __restrict__ boson,  // [bs, Ltau * Vs * 2] float32 
    const scalar_t* __restrict__ vec,     // [bs, Ltau * Vs] complex64
    scalar_t* __restrict__ out,           // [bs, Ltau * Vs] complex64
    const int Lx,  // typically Lx^2 = 10x10 = 100, up to 24x24 = 576
    const int Ltau, // typically 400, up to 24x40 = 960
    const int bs)
) {
    extern __shared__ scalar_t smem[];  // size: [Lx, Lx] * 2
    size_t smem_offset = 0;
    scalar_t* interm_vec_in = reinterpret_cast<scalar_t*>(smem + smem_offset);
    smem_offset += Lx * Lx * sizeof(scalar_t);
    scalar_t* interm_vec_out = reinterpret_cast<scalar_t*>(smem + smem_offset);
 
    // int Lx = blockDim.x;
    // int Ltau = gridDim.x;
    // int bs = gridDim.y;
    int stride_vs = Lx * Lx;
    int stride_tau_vs = stride_vs * Ltau;

    int tx = threadIdx.x;  
    int ty = threadIdx.y;
    int tau = blockIdx.x;
    int b = blockIdx.y;

    int bw = BLOCK_WIDTH;
    for (int offset_y = 0; offset_y < ceil_div(Lx, bw); offset_y++) {
        for (int offset_x = 0; offset_x < ceil_div(Lx, bw); offset_x++) {
            int global_x = offset_x * bw + tx;
            int global_y = offset_y * bw + ty;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }
            interm_vec_in[global_y * Lx + global_x] = vec[b * stride_tau_vs + tau * stride_vs + global_y * Lx + global_x];
            out[global_y * Lx + global_x] = vec[b * stride_tau_vs + tau * stride_vs + global_y * Lx + global_x];
        }
    }
    __syncthreads();


    // boson [Ltau, Ly, Lx, 2]
    // vec [Ltau, Ly, Lx]
    // center [Lx/2, Lx/2]
    // fam1
    int stride_tau_vs_2 = Ltau * Lx * Lx * 2;
    int stride_vs_2 = Lx * Lx * 2;
    int stride_lx_2 = Lx * 2;
    for (int cntr_offset_y = 0; cntr_offset_y < ceil_div(Lx/2, bw); cntr_offset_y++) {
        for (int cntr_offset_x = 0; cntr_offset_x < ceil_div(Lx/2, bw); cntr_offset_x++) {
            int cntr_x = cntr_offset_x * bw + tx;
            int cntr_y = cntr_offset_y * bw + ty;

            int global_y = cntr_y;
            int global_x = cntr_x + cntr_y % 2;
            if (global_x >= Lx || global_y >= Lx) {
                continue;  // Skip out-of-bound threads
            }         

            // fam1: x
            int idx_boson = bs * stride_tau_vs_2 + tau * stride_vs_2 + global_y * stride_lx_2 + global_x * 2 + 0;
            int i_vec = global_y * Lx + global_x;
            int j_vec = global_y * Lx + mod(global_x + 1, Lx) + Lx / 2;

            interm_vec_out[i_vec] = cosh(dtau) * interm_vec_in[i_vec] + sinh(dtau) * exp(1i * boson[idx_boson]);
            interm_vec_out[j_vec] = cosh(dtau) * interm_vec_in[j_vec] + sinh(dtau) * exp(-1i * boson[idx_boson]);
        }
    }
    __syncthreads();
} // mhm_vec_kernel
} // namespace cuda_pcg

torch::Tensor mhm_vec(
    // const torch::Tensor& i_lists, // [4, i_list], |i_list| = Vs/2
    // const torch::Tensor& j_lists, // [4, j_list]
    const torch::Tensor& boson,   // [bs, Ltau * Vs * 2] float32
    const torch::Tensor& vec,     // [bs, Ltau * Vs] complex64
    const int Lx,
) {
    TORCH_CHECK(vec.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(boson.is_cuda(), "Boson must  CUDA tensor");
    TORCH_CHECK(vec.scalar_type() == at::ScalarType::ComplexFloat, "Input tensor must be of type ComplexFloat");
    auto out = torch::empty_like(vec);
    auto bs = vec.size(0);  
    auto Lx = i_lists.size(1) * 2;
    auto Vs = Lx * Lx;
    auto Ltau = vec.size(1) / Vs; 

    auto out = torch::empty_like(vec);

    using scalar_t = cuFloatComplex;
    if (d_r.dtype() == at::ScalarType::ComplexFloat) {
        using scalar_t = cuFloatComplex; 
    } else if (d_r.dtype() == at::ScalarType::ComplexDouble) {
        using scalar_t = cuDoubleComplex;
    } else {
        throw std::invalid_argument("Unsupported data type");
    }

    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 grid(Ltau, bs);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cuda_pcg::mhm_vec_kernel<<<grid, block, 2 * Lx * Lx * sizeof(scalar_t), stream>>>(
        reinterpret_cast<float*>(boson.data_ptr()),
        reinterpret_cast<scalar_t*>(vec.data_ptr()),
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