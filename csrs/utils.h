#pragma once


__host__ __device__ inline int mod(int a, int b) {
    return (a % b + b) % b;
}

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}   