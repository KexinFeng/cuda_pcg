#include <torch/extension.h>
#include "cuda_pcg.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors", &add_tensors, "Add two tensors (CUDA)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pcg", &add_tensors, "Preconditioned Conjugate Gradient (CUDA)");
}
