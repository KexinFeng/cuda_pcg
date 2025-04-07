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