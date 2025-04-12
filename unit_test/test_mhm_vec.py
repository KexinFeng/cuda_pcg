import sys

sys.path.insert(0, '/users/4/fengx463/mount_folder/cuda_pcg')
sys.path.insert(0, '/users/4/fengx463/hmc/qed_fermion/qed_fermion')

import torch
from qed_fermion_module import _C
from hmc_sampler_batch import HmcSampler

# HMC inputs
Lx, Ly, Ltau = 2, 2, 4
Vs = Lx * Lx
print(f'max boson idx at tau=0, {Vs*2}')
hmc = HmcSampler(Lx=Lx, Ltau=Ltau)
hmc.dtau = 0.1
hmc.reset()

hmc.initialize_boson_test()

R_u = hmc.draw_psudo_fermion()  # cdtype

#------- batch_size = 1 -------     
psi_u = R_u.to(torch.complex64) # [bs, Ltau*Ly*Lx]
mat = hmc.get_diag_B_test(hmc.boson)
indices = mat.coalesce().indices()
values = mat.coalesce().values()
filtered_indices = indices[:, indices[0] < Lx * Ly]
filtered_values = values[indices[0] < Lx * Ly]
print("mat (COO format): filtered indices=", filtered_indices, "filtered values=", filtered_values)
print("psi_u:", psi_u.view(-1)[:Lx*Ly])

expected = torch.sparse.mm(mat, psi_u.T).to_dense()
expected = expected.T

#
boson = hmc.boson.permute(0, 4, 3, 2, 1).reshape(hmc.bs, -1).to(torch.float32).contiguous()
out = _C.mhm_vec(boson, psi_u, Lx, 0.1)

# Test if the output is close to the input
torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-8)
print("Success!")


# # #-------- batch_size > 1 -------
# bs = 2
# psi_u = R_u.to(torch.complex64).view(1, -1).repeat(bs, 1) # [bs, Ltau*Ly*Lx]
# expected = torch.sparse.mm(hmc.get_diag_B_test(hmc.boson), psi_u.T).to_dense()
# expected = expected.T

# #
# boson = hmc.boson.permute(0, 4, 3, 2, 1).reshape(hmc.bs, -1).to(torch.float32).contiguous()

# out = torch.empty_like(psi_u)
# for i in range(bs): 
#     out[i] = _C.mhm_vec(boson[i], psi_u[i], Lx, 0.1)

# # Test if the output is close to the input
# torch.testing.assert_close(out, psi_u, rtol=1e-5, atol=1e-8)
# print("Success!")
