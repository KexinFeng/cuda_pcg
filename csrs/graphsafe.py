import torch
from torch.utils.cpp_extension import load
mhm_mod = load(name="mhm_vec_mod", sources=["./csrs/mhm_vec_graphsafe.cu"], verbose=True)

Lx, Ltau, bs = 4, 4, 2
Vs = Lx * Lx
cdtype = torch.complex64

boson = torch.randn(bs, Ltau * Vs * 2, device='cuda', dtype=torch.float32)
vec = torch.randn(bs, Ltau * Vs, device='cuda', dtype=cdtype)

x = torch.ones_like(vec)

# Warm-up
for _ in range(3):
    out = mhm_mod.mhm_vec(boson, vec, Lx, 0.1, 8, 8)
    y = out + x

# Graph capture
vec_static = vec.clone()
graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph):
    out_graph = mhm_mod.mhm_vec(boson, vec_static, Lx, 0.1, 8, 8)
    y = out + x

graph.replay()
print(out_graph[0, :5])
