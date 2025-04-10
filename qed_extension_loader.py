from torch.utils.cpp_extension import load
import os
import torch
base_dir = os.path.dirname(os.path.abspath(__file__))

CUDA_HOME = os.environ.get("CUDA_HOME", "/common/software/install/manual/cuda/12.0")

# FLAGS
CXX_FLAGS = ["-g", "-O1", "-fPIC", "-std=c++17"]
NVCC_FLAGS = ["-O1", "-lineinfo", "-Xcompiler", "-fPIC", "-std=c++17"]

# CXX11 ABI
USE_CXX11_ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={USE_CXX11_ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={USE_CXX11_ABI}"]

# Threads
nvcc_threads = int(os.getenv("NVCC_THREADS", 8))
num_threads = min(os.cpu_count(), nvcc_threads)
NVCC_FLAGS += ["--threads", str(num_threads)]

_C = load(
    name="qed_fermion_module",  # Python module name
    sources=[
        # "csrs/cuda_pcg_cusparse.cu",  # optional if you use it
        base_dir + "/csrs/precon_vec.cu",
        base_dir + "/csrs/pybind.cpp",
    ],
    extra_include_paths=[
        os.path.join(CUDA_HOME, "include"),
    ],
    extra_ldflags=[
        f"-L{os.path.join(CUDA_HOME, 'lib64')}",
        # "-lcudart", "-lcusparse", "-lcublas",
    ],
    extra_cflags=CXX_FLAGS,
    extra_cuda_cflags=NVCC_FLAGS,
    verbose=True,
)