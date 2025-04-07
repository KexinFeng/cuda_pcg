from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_modules = []

extension_sources = [
    "csrs/cuda_pcg.cu",
    "csrs/pybind.cpp",
]

extension = CUDAExtension(
    name="qed_fermion._C",  # output name
    sources=extension_sources,
    # extra_compile_args={
    #     "cxx": CXX_FLAGS,
    #     "nvcc": NVCC_FLAGS,
    # },
)
ext_modules.append(extension)

setup(
    name='Awsome_Project',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
