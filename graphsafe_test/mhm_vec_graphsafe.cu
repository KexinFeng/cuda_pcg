#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/complex.h>
#include "device_func.cu"
  
torch::Tensor mhm_vec2(
    const torch::Tensor& boson,   // [bs, Ltau * Vs * 2] float32
    const torch::Tensor& vec,     // [bs, Ltau * Vs] complex64
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
    torch::Tensor out1 = torch::empty_like(vec);
    torch::Tensor out2 = torch::empty_like(vec);

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
    cudaStreamCaptureStatus capture_status;
    cudaStreamIsCapturing(stream, &capture_status);

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

    if (capture_status != cudaStreamCaptureStatusActive &&
    capture_status != cudaStreamCaptureStatusInvalidated) {
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "CUDA stream synchronization failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA kernel execution failed");
        }
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
    if (capture_status != cudaStreamCaptureStatusActive &&
    capture_status != cudaStreamCaptureStatusInvalidated) {
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "CUDA stream synchronization failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA kernel execution failed");
        }
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
    if (capture_status != cudaStreamCaptureStatusActive &&
    capture_status != cudaStreamCaptureStatusInvalidated) {
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "CUDA stream synchronization failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA kernel execution failed");
        }
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
    if (capture_status != cudaStreamCaptureStatusActive &&
    capture_status != cudaStreamCaptureStatusInvalidated) {
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "CUDA stream synchronization failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA kernel execution failed");
        }
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mhm_vec", &mhm_vec, "MHM vector (CUDA)");
    m.def("mhm_vec2", &mhm_vec2, "MHM vector (CUDA)");
}
