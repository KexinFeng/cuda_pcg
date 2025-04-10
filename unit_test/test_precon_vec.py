import sys

sys.path.insert(0, '/users/4/fengx463/mount_folder/cuda_pcg')
sys.path.insert(0, '/users/4/fengx463/hmc/qed_fermion/qed_fermion')

import torch
from qed_fermion_module import _C
from hmc_sampler_batch import HmcSampler
# from qed_extension_loader import _C

# HMC inputs
Lx, Ly, Ltau = 4, 4, 10
Vs = Lx * Lx
hmc = HmcSampler(Lx=Lx, Ltau=Ltau)
boson = hmc.boson  # dtype

R_u = hmc.draw_psudo_fermion().view(-1, 1)  # cdtype
psi_u = R_u.to(torch.complex64)

hmc.reset_precon()
precon = hmc.precon.to_sparse_csr().to(torch.complex64)

out = _C.precon_vec(psi_u, 
                precon,
                Lx)
print("Result of PCG:", out[-10:], out.shape)

expected = torch.sparse.mm(precon, psi_u).to_dense()
print("Expected result:", expected[-10:], expected.shape)
torch.testing.assert_close(expected, out, atol=1e-6, rtol=0)






