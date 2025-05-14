#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuComplex.h>
#include <iostream>

__global__ void dummy_mhm_vec_kernel(const float* __restrict__ boson,
                                     const cuFloatComplex* __restrict__ vec,
                                     cuFloatComplex* __restrict__ out,
                                     int Lx, float dtau, int tau_roll) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Lx) {
        float factor = 1.0f + dtau * tau_roll;
        out[idx] = cuCmulf(vec[idx], make_cuFloatComplex(factor, factor));
    }
}

torch::Tensor mhm_vec(torch::Tensor boson, 
                      torch::Tensor vec,
                      int64_t Lx, 
                      float dtau,
                      int64_t block_size_x = 8, 
                      int64_t block_size_y = 8) {
    TORCH_CHECK(boson.dim() == 2, "boson must be [bs, Ltau * Vs * 2]");
    TORCH_CHECK(vec.dim() == 2, "vec must be [bs, Ltau * Vs]");
    TORCH_CHECK(vec.scalar_type() == torch::kComplexFloat);
    TORCH_CHECK(boson.scalar_type() == torch::kFloat32);
    TORCH_CHECK(boson.is_cuda() && vec.is_cuda());

    auto out = torch::empty_like(vec);

    int64_t bs = vec.size(0);
    int64_t Vs = Lx * Lx;
    int64_t Ltau = vec.size(1) / Vs;

    dim3 block(block_size_x * block_size_y);
    dim3 grid((vec.numel() + block.x - 1) / block.x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    for (int tau_roll = 0; tau_roll < 2; ++tau_roll) {
        dummy_mhm_vec_kernel<<<grid, block, 0, stream>>>(
            boson.data_ptr<float>(),
            reinterpret_cast<cuFloatComplex*>(vec.data_ptr()),
            reinterpret_cast<cuFloatComplex*>(out.data_ptr()),
            Lx, dtau, tau_roll
        );
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mhm_vec", &mhm_vec, "Mock MHM Vec with CUDA Graph compatibility");
}
