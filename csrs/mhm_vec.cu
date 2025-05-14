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

torch::Tensor mhm_vec2(
    const torch::Tensor& boson,   // [bs, Ltau * Vs * 2] float32
    const torch::Tensor& vec,     // [bs, Ltau * Vs] complex64
    torch::Tensor& out1,
    torch::Tensor& out2,
    const int64_t Lx,
    const float dtau,
    const int64_t block_size_x = 8,
    const int64_t block_size_y = 8)
{
    TORCH_CHECK(boson.dim() == 2, "Boson tensor must have 2 dimensions: [bs, Ltau * Vs * 2]");
    TORCH_CHECK(vec.dim() == 2, "Input tensor must have 2 dimensions: [bs, Ltau * Vs]");
    TORCH_CHECK(boson.size(0) == vec.size(0), "Batch size of boson and vec tensors must match");
    TORCH_CHECK(boson.size(1) == vec.size(1) * 2, "Boson tensor's second dimension must be twice the size of vec's second dimension");

    TORCH_CHECK(vec.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(boson.is_cuda(), "Boson must  CUDA tensor");
    TORCH_CHECK(vec.scalar_type() == at::ScalarType::ComplexFloat, "Input tensor must be of type ComplexFloat");
    TORCH_CHECK(boson.scalar_type() == at::ScalarType::Float, "Boson tensor must be of type Float");

    torch::Tensor vec_in = vec;
    // torch::Tensor out1 = torch::empty_like(vec);
    // torch::Tensor out2 = torch::empty_like(vec);

    int64_t bs = vec.size(0);
    int64_t Vs = Lx * Lx;
    int64_t Ltau = vec.size(1) / Vs; 

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

    dim3 block = {block_size_x, block_size_y};
    dim3 grid = {Ltau, bs};
    int64_t tau_roll = 0;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // cudaStreamCaptureStatus capture_status;
    // cudaStreamIsCapturing(stream, &capture_status);

    // B_vec_mul
    cuda_pcg::mhm_vec_kernel<<<grid, block, 2 * Vs * sizeof(scalar_t), stream>>>(
        reinterpret_cast<float*>(boson.data_ptr()),
        reinterpret_cast<scalar_t*>(vec_in.data_ptr()),
        reinterpret_cast<scalar_t*>(out1.data_ptr()),
        Lx, dtau, tau_roll);
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

    // B_vec_mul
    tau_roll = 1;
    cuda_pcg::mhm_vec_kernel<<<grid, block, 2 * Vs * sizeof(scalar_t), stream>>>(
        reinterpret_cast<float*>(boson.data_ptr()),
        reinterpret_cast<scalar_t*>(vec_in.data_ptr()),
        reinterpret_cast<scalar_t*>(out1.data_ptr()),
        Lx, dtau, tau_roll);
        
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

    return out2;      
}

torch::Tensor mh_vec(
    const torch::Tensor& boson,   // [bs, Ltau * Vs * 2] float32
    const torch::Tensor& vec,     // [bs, Ltau * Vs] complex64
    torch::Tensor& out1,
    torch::Tensor& out2,
    const int64_t Lx,
    const float dtau,
    const int64_t block_size_x = 8,
    const int64_t block_size_y = 8)
{
    TORCH_CHECK(boson.dim() == 2, "Boson tensor must have 2 dimensions: [bs, Ltau * Vs * 2]");
    TORCH_CHECK(vec.dim() == 2, "Input tensor must have 2 dimensions: [bs, Ltau * Vs]");
    TORCH_CHECK(boson.size(0) == vec.size(0), "Batch size of boson and vec tensors must match");
    TORCH_CHECK(boson.size(1) == vec.size(1) * 2, "Boson tensor's second dimension must be twice the size of vec's second dimension");

    TORCH_CHECK(vec.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(boson.is_cuda(), "Boson must  CUDA tensor");
    TORCH_CHECK(vec.scalar_type() == at::ScalarType::ComplexFloat, "Input tensor must be of type ComplexFloat");
    TORCH_CHECK(boson.scalar_type() == at::ScalarType::Float, "Boson tensor must be of type Float");

    torch::Tensor vec_in = vec;
    // torch::Tensor out1 = torch::empty_like(vec);
    // torch::Tensor out2 = torch::empty_like(vec);

    int64_t bs = vec.size(0);
    int64_t Vs = Lx * Lx;
    int64_t Ltau = vec.size(1) / Vs; 

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

    dim3 block = {block_size_x, block_size_y};
    dim3 grid = {Ltau, bs};
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();


    // B_vec_mul
    int64_t tau_roll = 1;
    cuda_pcg::mhm_vec_kernel<<<grid, block, 2 * Vs * sizeof(scalar_t), stream>>>(
        reinterpret_cast<float*>(boson.data_ptr()),
        reinterpret_cast<scalar_t*>(vec_in.data_ptr()),
        reinterpret_cast<scalar_t*>(out1.data_ptr()),
        Lx, dtau, tau_roll);
        
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

    return out2; 
}

__global__ void dummy_mhm_vec_kernel(const float* __restrict__ boson,
                                     const cuFloatComplex* __restrict__ vec,
                                     cuFloatComplex* __restrict__ out,
                                     int Lx, float dtau, int tau_roll) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Lx) {
        float val = boson[i] + dtau + tau_roll;
        out[i] = make_cuFloatComplex(val, val);
    }
}

__global__ void dummy_vec_sub_kernel(const cuFloatComplex* a, const cuFloatComplex* b, cuFloatComplex* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = cuCsubf(a[i], b[i]);
    }
}

torch::Tensor mhm_vec(
    const torch::Tensor& boson,   // [bs, Ltau * Vs * 2] float32
    const torch::Tensor& vec,     // [bs, Ltau * Vs] complex64
    torch::Tensor& out1,
    torch::Tensor& out2,
    const int64_t Lx,
    const float dtau,
    const int64_t block_size_x = 8,
    const int64_t block_size_y = 8)
{
    TORCH_CHECK(boson.dim() == 2);
    TORCH_CHECK(vec.dim() == 2);
    TORCH_CHECK(boson.size(1) == vec.size(1) * 2);

    int64_t bs = vec.size(0);
    int64_t Vs = Lx * Lx;
    int64_t Ltau = vec.size(1) / Vs;
    dim3 block(block_size_x, block_size_y);
    dim3 grid((Ltau + block.x - 1) / block.x, bs);

    // torch::Tensor out1 = torch::empty_like(vec);
    // torch::Tensor out2 = torch::empty_like(vec);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dummy_mhm_vec_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(boson.data_ptr()),
        reinterpret_cast<const cuFloatComplex*>(vec.data_ptr()),
        reinterpret_cast<cuFloatComplex*>(out1.data_ptr()),
        Lx, dtau, 0);

    dummy_vec_sub_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const cuFloatComplex*>(vec.data_ptr()),
        reinterpret_cast<const cuFloatComplex*>(out1.data_ptr()),
        reinterpret_cast<cuFloatComplex*>(out2.data_ptr()),
        vec.numel());

    dummy_mhm_vec_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(boson.data_ptr()),
        reinterpret_cast<const cuFloatComplex*>(out2.data_ptr()),
        reinterpret_cast<cuFloatComplex*>(out1.data_ptr()),
        Lx, dtau, 1);

    dummy_vec_sub_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const cuFloatComplex*>(out2.data_ptr()),
        reinterpret_cast<const cuFloatComplex*>(out1.data_ptr()),
        reinterpret_cast<cuFloatComplex*>(out2.data_ptr()),
        vec.numel());

    return out2;
}