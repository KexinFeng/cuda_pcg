# README
In this repo, efficient cuda kernels for efficient matrix-vector multiplication is built, which is applied to preconditioned cg algorithm in the context of quantum electrodynamics and fermion coupling. A shared memroy has been used properly in order to reduce the HBM and SRAM communication. **>3x** speedup is achieved, where the baseline is naive gpu usage meaning simply specifying device='cuda'.

## Latency Speedup

![Speedup Plot](illustration/spdup.png)

