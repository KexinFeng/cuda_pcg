#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#include <torch/extension.h>
#include "cuda_pcg.h"


// Error-checking macros
#define CHECK_CUDA(call) {                                                 \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(err));                                  \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
}

#define CHECK_CUSPARSE(call) {                                             \
    cusparseStatus_t status = call;                                        \
    if (status != CUSPARSE_STATUS_SUCCESS) {                               \
        fprintf(stderr, "CUSPARSE error %s:%d: %d\n", __FILE__, __LINE__,   \
                status);                                                   \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
}

#define CHECK_CUBLAS(call) {                                               \
    cublasStatus_t status = call;                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                 \
        fprintf(stderr, "CUBLAS error %s:%d: %d\n", __FILE__, __LINE__,     \
                status);                                                   \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
}


__global__ void add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    auto output = torch::zeros_like(a);
    int size = a.numel();

    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}

__global__ void pcg_kernel(const float* A, const float* b, float* x, int n, int max_iter) {
    // Placeholder for PCG algorithm implementation
    // This is where the actual PCG algorithm would be implemented
    // For now, we just copy b to x as a placeholder
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = b[idx];
    }
}

torch::Tensor pcg(torch::Tensor A, torch::Tensor b, int max_iter) {
    int n = A.size(0);
    auto x = torch::zeros_like(b);

    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;

    pcg_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        b.data_ptr<float>(),
        x.data_ptr<float>(),
        n,
        max_iter
    );

    return x;
}

torch::Tensor pcg(
    torch::Tensor i_lists,  // [4, Vs/2] start site of each family
    torch::Tensor j_lists,  // [4, Vs/2] end site of each family
    torch::Tensor boson,    // [bs, Ltau * Vs * 2] float32, U(1) configuration
    torch::Tensor psi_u,    // [bs, Ltau * Vs] complex64
    torch::Tensor precon,   // [Ltau * Vs, Ltau * Vs] complex64, sparse_csr
    int Lx, 
    int Ltau, 
    int max_iter, 
    double rtol) 
{
    // Ensure `precon` is a sparse CSR tensor
    TORCH_CHECK(precon.is_sparse_csr(), "precon must be a torch.sparse.csr tensor");

    // --- Create cuSPARSE and cuBLAS handles and CUDA stream ---
    cusparseHandle_t cusparseHandle;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, stream));
    CHECK_CUBLAS(cublasSetStream(cublasHandle, stream));

    // --- Convert `precon` to cuSPARSE CSR format ---
    auto precon_crow_indices = precon.crow_indices();
    auto precon_col_indices = precon.col_indices();
    auto precon_values = precon.values();

    const int64_t* d_crow_indices = precon_crow_indices.data_ptr<int64_t>();
    const int64_t* d_col_indices = precon_col_indices.data_ptr<int64_t>();
    const cuDoubleComplex* d_values = reinterpret_cast<const cuDoubleComplex*>(precon_values.data_ptr<std::complex<double>>());

    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA,
        precon.size(0), precon.size(1), precon_values.size(0),
        const_cast<int64_t*>(d_crow_indices),
        const_cast<int64_t*>(d_col_indices),
        const_cast<cuDoubleComplex*>(d_values),
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));

    // --- Convert `psi_u` to cuSPARSE dense vector format ---
    const cuDoubleComplex* d_p = reinterpret_cast<const cuDoubleComplex*>(psi_u.data_ptr<std::complex<double>>());

    cusparseDnVecDescr_t vecP, vecAp;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, psi_u.numel(), const_cast<cuDoubleComplex*>(d_p), CUDA_C_64F));

    // Allocate memory for Ap (output of SpMV)
    auto Ap = torch::zeros_like(psi_u);
    cuDoubleComplex* d_Ap = reinterpret_cast<cuDoubleComplex*>(Ap.data_ptr<std::complex<double>>());
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, psi_u.numel(), d_Ap, CUDA_C_64F));

    // --- Allocate an external buffer for cusparseSpMV ---
    size_t bufferSize = 0;
    void* dBuffer = NULL;
    cuDoubleComplex spmvAlpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex spmvBeta  = make_cuDoubleComplex(0.0, 0.0);
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmvAlpha, matA, vecP, &spmvBeta, vecAp,
        CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // --- Perform the SpMV operation ---
    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmvAlpha, matA, vecP, &spmvBeta, vecAp,
        CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    // Return the result of SpMV (Ap)
    return Ap;
}
