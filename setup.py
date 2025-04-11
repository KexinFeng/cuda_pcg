from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

CUDA_HOME = os.environ.get("CUDA_HOME", "/common/software/install/manual/cuda/12.0")

# FLAGS
CXX_FLAGS = ["-O0", "-g", "-fPIC", "-std=c++17"]
NVCC_FLAGS = ["-O0", "-g", "-Xcompiler", "-fPIC", "-std=c++17"]

# CXX11 ABI
USE_CXX11_ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={USE_CXX11_ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={USE_CXX11_ABI}"]

# Use NVCC threads to parallelize the build.
nvcc_threads = int(os.getenv("NVCC_THREADS", 4))
num_threads = min(len(os.sched_getaffinity(0))//2, nvcc_threads)
NVCC_FLAGS += ["--threads", str(num_threads)]

extension = CUDAExtension(
    name="qed_fermion_module._C",  # the Python import name
    sources=[
        # "csrs/cuda_pcg_cusparse.cu",
        # "csrs/precon_vec.cu",
        "csrs/mhm_vec.cu",
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

setup(
    name="Awsome_Project",
    version="0.1",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
)
