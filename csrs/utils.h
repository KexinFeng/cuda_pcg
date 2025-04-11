#pragma once


__host__ __device__ inline int mod(int a, int b) {
    return (a % b + b) % b;
}

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}   

// overloading operators
__device__ __host__ cuFloatComplex  operator+(cuFloatComplex a, cuFloatComplex b) { return cuCaddf(a,b); }
__device__ __host__ cuFloatComplex  operator-(cuFloatComplex a, cuFloatComplex b) { return cuCsubf(a,b); }
__device__ __host__ cuFloatComplex  operator*(cuFloatComplex a, cuFloatComplex b) { return cuCmulf(a,b); }
__device__ __host__ cuFloatComplex  operator/(cuFloatComplex a, cuFloatComplex b) { return cuCdivf(a,b); }
