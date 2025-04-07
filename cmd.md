python setup.py build_ext --inplace
python setup.py clean --all
python setup.py install

Clean old build
python setup.py clean --all
rm -rf build/ tensor_ops/*.so tensor_ops/*.cpp tensor_ops/*.cu.o

Check installation
python -c "import torch; print(torch.version.cuda)"
python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"


🧪 Extra Tip: Use torch.utils.cpp_extension.load() for live loading (no setup.py)
Replace setup-based build with inline compilation in main.py:

python
Copy
Edit
from torch.utils.cpp_extension import load

tensor_add_cuda = load(
    name="tensor_add_cuda",
    sources=["tensor_ops/tensor_add.cpp", "tensor_ops/tensor_add_cuda.cu"],
    verbose=True
)
This way, you don’t need to run setup.py every time. Great for debugging!


nvcc -o pcg_cuda cuda_pcg.cu -lcusparse -lcublas

nvcc -o pcg_cuda cuda_pcg.cu -lcusparse -lcublas -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include -L/users/4/fengx463/.local/lib/python3.9/site-packages/torch/lib -ltorch -ltorch_cpu

nvcc -o pcg_cuda cuda_pcg.cu -lcusparse -lcublas -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -L/users/4/fengx463/.local/lib/python3.9/site-packages/torch/lib -ltorch -ltorch_cpu

nvcc -o pcg_cuda cuda_pcg.cu \
    -lcusparse -lcublas \
    -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include \
    -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include \
    -L/users/4/fengx463/.local/lib/python3.9/site-packages/torch/lib \
    -ltorch -ltorch_cpu \
    --std=c++17 \
    -I/common/software/install/migrated/anaconda/python3-2021.11-mamba/include/python3.9

nvcc -o cuda_pcg.o -c csrs/cuda_pcg.cu \
    -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include \
    -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include \
    -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/THC \
    -I/common/software/install/manual/cuda/12.0/include \
    -I/common/software/install/migrated/anaconda/python3-2021.11-mamba/include/python3.9 \
    --compiler-options '-fPIC' --std=c++17

# Step 1: Compile cuda_pcg.cu into an object file
nvcc -c -o cuda_pcg.o cuda_pcg.cu \
    -I/path/to/libtorch/include \
    -I/path/to/libtorch/include/torch/csrc/api/include \
    -L/path/to/libtorch/lib \
    -lcusparse \
    -lcublas \
    -lcudart \
    -std=c++17

# Step 2: Compile main.cpp and link with cuda_pcg.o
g++ -o main csrs/main.cpp cuda_pcg.o \
    -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include \
    -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include \
    -L/users/4/fengx463/.local/lib/python3.9/site-packages/torch/lib \
    -I/common/software/install/manual/cuda/12.0/include \
    -I/common/software/install/migrated/anaconda/python3-2021.11-mamba/include/python3.9 \
    -ltorch \
    -ltorch_cpu \
    -lc10 \
    -lcusparse \
    -lcublas -lcudart -std=c++17



