#pragma once


__host__ __device__ inline int mod(int a, int b) {
    return (a % b + b) % b;
}