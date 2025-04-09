import sys

sys.path.insert(0, '/users/4/fengx463/mount_folder/cuda_pcg')
sys.path.insert(0, '/users/4/fengx463/hmc/qed_fermion/qed_fermion')

import torch
import qed_fermion_module._C as core
from hmc_sampler_batch import HmcSampler

from torch.utils.cpp_extension import load
import torch
# Load the CUDA extension
_C = load(
    name="_C",
    sources=["csrs/pybind.cpp", "csrs/precon_vec.cu"],
    verbose=True
)

# HMC inputs
Lx, Ly, Ltau = 4, 4, 40
hmc = HmcSampler(Lx=Lx, Ltau=Ltau)
boson = hmc.boson  # dtype

R_u = hmc.draw_psudo_fermion().view(-1, 1)  # cdtype
psi_u = R_u.to(torch.complex64)

hmc.reset_precon()
precon = hmc.precon.to_sparse_csr().to(torch.complex64)
# out = torch.zeros_like(psi_u, dtype=torch.complex64)

out = _C.precon_vec(psi_u, 
                precon,
                Lx)
print("Result of PCG:", out[:10], out.shape)

# expected = torch.sparse.mm(precon, psi_u)
# print("Expected result:", expected[:10])
# torch.testing.assert_close(expected, out, atol=1e-6, rtol=0, msg="The output does not match the expected result.")




