import torch
from torch.utils.cpp_extension import load
_C = load(
    name="_C", 
    sources=["./graphsafe_test/device_func.cu",
             "./graphsafe_test/mhm_vec_graphsafe.cu", 
             "./graphsafe_test/utils.h"], 
    verbose=True)

Lx, Ltau, bs = 4, 4, 2
Ly = Lx
Vs = Lx * Lx
cdtype = torch.complex64

boson = torch.randn(bs, Ltau * Vs * 2, device='cuda', dtype=torch.float32)
vec = torch.randn(bs, Ltau * Vs, device='cuda', dtype=cdtype)

BLOCK_SIZE = (4, 8)

out1 = torch.empty(bs, Lx * Ly * Ltau, dtype=cdtype, device='cuda')
out2 = torch.empty(bs, Lx * Ly * Ltau, dtype=cdtype, device='cuda')


x = torch.ones_like(vec)

# Warm-up
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        tmp = _C.mhm_vec2(boson, vec, out1, out2, Lx, Ltau, *BLOCK_SIZE)
        x.copy_(tmp)
    s.synchronize()
torch.cuda.current_stream().wait_stream(s)

# Graph capture
vec_static = vec.clone()
boson_static = boson.clone()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out_graph = _C.mhm_vec2(boson_static, vec_static, out1, out2, Lx, Ltau, *BLOCK_SIZE)
    x.copy_(tmp)

# Graph relay
graph.replay()
print(out_graph[0, :5])
