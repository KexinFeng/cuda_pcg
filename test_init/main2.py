from torch.utils.cpp_extension import load
import torch
# Load the CUDA extension
_C = load(
    name="_C",
    sources=["csrs/pybind.cpp", 
             "csrs/cuda_pcg.cu"],
    verbose=True
)

a = torch.randn(3, 3, device='cuda')
b = torch.randn(3, 3, device='cuda')

print("Tensor A:", a)
print("Tensor B:", b)

result = _C.add_tensors(a, b)
print("Result device:", result.device)
print("Result (A + B):", result)
