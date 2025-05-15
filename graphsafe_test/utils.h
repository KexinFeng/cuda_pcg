#pragma once


__host__ __device__ inline int mod(int a, int b) {
    return (a % b + b) % b;
}

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}   

// overloading operators
__device__ __host__ inline cuFloatComplex  operator+(cuFloatComplex a, cuFloatComplex b) { return cuCaddf(a,b); }
__device__ __host__ inline cuFloatComplex  operator-(cuFloatComplex a, cuFloatComplex b) { return cuCsubf(a,b); }
__device__ __host__ inline cuFloatComplex  operator*(cuFloatComplex a, cuFloatComplex b) { return cuCmulf(a,b); }
__device__ __host__ inline cuFloatComplex  operator/(cuFloatComplex a, cuFloatComplex b) { return cuCdivf(a,b); }

  

template<typename scalar_t>
__device__ void mat_vec_mul_2b2(    
    const float* __restrict__ boson,
    scalar_t* __restrict__ interm_vec_in,
    scalar_t* __restrict__ interm_vec_out,
    int64_t idx_boson,
    int64_t i_vec,
    int64_t j_vec,
    float dtau_factor) 
{
    // vec_out[i_vec, j_vec] = mat @ vec_in[i_vec, j_vec]                  
    float boson_val = boson[idx_boson];
    cuFloatComplex cosh_dtau = make_cuFloatComplex(coshf(dtau_factor), 0.0f);
    cuFloatComplex sinh_dtau = make_cuFloatComplex(sinhf(dtau_factor), 0.0f);
    float cos_boson = cosf(boson_val);
    float sin_boson = sinf(boson_val);
    cuFloatComplex sinh_exp_pos = sinh_dtau * make_cuFloatComplex(cos_boson, sin_boson);  // exp(1i * boson_val)
    cuFloatComplex sinh_exp_neg = sinh_dtau * make_cuFloatComplex(cos_boson, -sin_boson); // exp(-1i * boson_val)

    interm_vec_out[i_vec] = cosh_dtau * interm_vec_in[i_vec] + sinh_exp_pos * interm_vec_in[j_vec];
    interm_vec_out[j_vec] = cosh_dtau * interm_vec_in[j_vec] + sinh_exp_neg * interm_vec_in[i_vec];
}
