import time
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

CUDA_HOME = os.environ.get("CUDA_HOME", "/common/software/install/manual/cuda/12.0")

# FLAGS
# CXX_FLAGS = ["-O0", "-g", "-fPIC", "-std=c++17"]
CXX_FLAGS = ["-O1", "-fPIC", "-std=c++17"]
# NVCC_FLAGS = ["-O1", "-lineinfo", "-Xcompiler", "-fPIC", "-std=c++17"]
# NVCC_FLAGS = ["-O0", "-g", "-G", "-Xcompiler", "-fPIC", "-std=c++17"]
NVCC_FLAGS = ["-O1", "-Xcompiler", "-fPIC", "-std=c++17"]  
# -g +9s -O0,O2,O3 +3s
# total 47s for [3/3] files

# CXX11 ABI
USE_CXX11_ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={USE_CXX11_ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={USE_CXX11_ABI}"]

# Use NVCC threads to parallelize the build.
nvcc_threads = int(os.getenv("NVCC_THREADS", 8))
num_threads = min(len(os.sched_getaffinity(0)) - 2, nvcc_threads)  
# os.sched_getaffinity(0) = num_cores + 2; ninja -j N  N_default=os.sched_getaffinity(0) + 2
NVCC_FLAGS += ["--threads", str(num_threads)]

extension = CUDAExtension(
    name="qed_fermion_module._C",  # the Python import name
    sources=[
        # "csrs/cuda_pcg_cusparse.cu",
        # "csrs/precon_vec.cu",
        # "csrs/mhm_vec.cu",
        "csrs/b_vec_per_tau.cu",
        "csrs/pybind.cpp",
    ],
    include_dirs=[
        os.path.join(CUDA_HOME, "include"),
    ],
    library_dirs=[
        os.path.join(CUDA_HOME, "lib64"),
    ],
    # libraries=["cudart", "cusparse", "cublas"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)

# Measure compilation time
start_time = time.time()  # Start timer

setup(
    name="Awsome_Project",
    version="0.1",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
)
end_time = time.time()  # End timer
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Compilation Time: {elapsed_time:.2f} seconds")