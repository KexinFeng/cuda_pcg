#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/complex.h>
#include "cuda_pcg.h"
#include "utils.h"


#define BLOCK_SIZE 64  // 8x8, limit 1024: 32x32, 512: 24x24, 256: 16x16, 128: 10x10

torch::Tensor O_vec(
    const torch::Tensor& i_lists, // [4, i_list], |i_list| = Vs/2
    const torch::Tensor& j_lists, // [4, j_list]
    const torch::Tensor& boson,   // [bs, Ltau * Vs * 2] float32
    const torch::Tensor& vec,     // [bs, Ltau * Vs] complex64
) {
    TORCH_CHECK(vec.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(boson.is_cuda(), "Boson must be a CUDA tensor");
    TORCH_CHECK(vec.scalar_type() == at::ScalarType::ComplexFloat, "Input tensor must be of type ComplexFloat");
    auto out = torch::empty_like(vec);
    auto bs = vec.size(0);  
    auto Lx = i_lists.size(1) * 2;
    auto Vs = Lx * Lx;
    auto Ltau = vec.size(1) / Vs; 

    dim3 block(BLOCK_WIDTH, 1, 1);
}