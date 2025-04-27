# README
In this repo, efficient cuda kernels for efficient matrix-vector multiplication is built, which is applied to preconditioned cg algorithm in the context of quantum electrodynamics and fermion coupling. A shared memroy has been used properly in order to reduce the HBM and SRAM communication. **>3x** speedup is achieved, where the baseline is naive gpu usage meaning simply specifying device='cuda'.


[latency_speedup.pdf](https://github.com/user-attachments/files/19930832/latency_speedup.pdf)
