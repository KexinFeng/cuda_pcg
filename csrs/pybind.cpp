#include <torch/extension.h>
#include "cuda_pcg.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("add_tensors", &add_tensors, "Add two tensors (CUDA)");
    // m.def("pcg", &pcg, "Preconditioned Conjugate Gradient (CUDA)");
//     m.def("precon_vec", &precon_vec, "Preconditioner vector (CUDA)");
    m.def("mhm_vec", &mhm_vec, "MHM vector (CUDA)",
          py::arg("boson"),
          py::arg("vec"),
          py::arg("Lx"),
          py::arg("dtau"));
    m.def("b_vec_per_tau", &b_vec_per_tau, "B vector (CUDA)",
          py::arg("boson"),
          py::arg("vec"),
          py::arg("Lx"),
          py::arg("dtau"),
          py::arg("interm_out_bool"));
}
