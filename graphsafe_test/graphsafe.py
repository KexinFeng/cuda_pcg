import torch
from torch.utils.cpp_extension import load
import os
import time
import shutil
import glob

# Measure compilation time
start_time = time.time()  # Start timer

# Loader
CUDA_HOME = os.environ.get("CUDA_HOME", "/common/software/install/manual/cuda/12.0")

# FLAGS
# CXX_FLAGS = ["-O0", "-g", "-fPIC", "-std=c++17"]
CXX_FLAGS = ["-O3", "-fPIC", "-std=c++17"]
# NVCC_FLAGS = ["-O1", "-lineinfo", "-Xcompiler", "-fPIC", "-std=c++17"]
# NVCC_FLAGS = ["-O0", "-g", "-G", "-Xcompiler", "-fPIC", "-std=c++17"]
NVCC_FLAGS = ["-O3", "-Xcompiler", "-fPIC", "-std=c++17"]  
# -g +9s -O0,O2,O3 +3s
# total 47s for [3/3] files

# CXX11 ABI
USE_CXX11_ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={USE_CXX11_ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={USE_CXX11_ABI}"]

# Use NVCC threads to parallelize the build.
nvcc_threads = int(os.getenv("NVCC_THREADS", 8))
num_threads = max(2, min(len(os.sched_getaffinity(0)) - 2, nvcc_threads))
num_threads = 1
print(f'----->{num_threads}<------') 
# os.sched_getaffinity(0) = num_cores + 2; ninja -j N  N_default=os.sched_getaffinity(0) + 2
NVCC_FLAGS += ["--threads", str(num_threads)]

os.makedirs("./graphsafe_test/build", exist_ok=True)

_C = load(
    name="_C", 
    sources=["./graphsafe_test/device_func.cu",
             "./graphsafe_test/mhm_vec_graphsafe.cu"], 
    extra_include_paths=[
        os.path.join(CUDA_HOME, "include"),
    ],
    extra_cflags=CXX_FLAGS,
    extra_cuda_cflags=NVCC_FLAGS,
    verbose=True,
    is_python_module=True,
)

end_time = time.time()  # End timer
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Compilation Time: {elapsed_time:.2f} seconds")

so_files = glob.glob(os.path.expanduser("~/.cache/torch_extensions/py39_cu121/_C/_C.so"))
dst_dir = os.path.expanduser("~/hmc/qed_fermion/qed_fermion/_C.cpython-39-x86_64-linux-gnu.so")
for so_file in so_files:
    shutil.copy(so_file, dst_dir)
print(f"Copied {len(so_files)} .so files to {dst_dir}")

# Main
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
        tmp = _C.mhm_vec(boson, vec, Lx, Ltau, *BLOCK_SIZE)
        x.copy_(tmp)
    s.synchronize()
torch.cuda.current_stream().wait_stream(s)

# Graph capture
vec_static = vec.clone()
boson_static = boson.clone()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out_graph = _C.mhm_vec(boson_static, vec_static, Lx, Ltau, *BLOCK_SIZE)
    x.copy_(tmp)

# Graph relay
graph.replay()
print(out_graph[0, :5])
