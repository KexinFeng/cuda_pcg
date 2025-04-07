from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_modules = []

extension_sources = [
    "csrs/cuda_pcg.cu",
    "csrs/pybind.cpp",
]

extension = CUDAExtension(
    name="qed_fermion_module._C",  # output name
    sources=extension_sources,
    extra_compile_args={
        "cxx": [],  # Add any C++ specific flags if needed
        "nvcc": [
            "-I/common/software/install/manual/cuda/12.0/include",  # Include CUDA headers
        ],
    },
    libraries=["cudart", "cusparse", "cublas"],  # Link against CUDA libraries
    library_dirs=["/common/software/install/manual/cuda/12.0/lib64"],  # Path to CUDA libraries
)
ext_modules.append(extension)

setup(
    name='Awsome_Project',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
