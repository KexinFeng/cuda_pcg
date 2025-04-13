import sys

sys.path.insert(0, '/users/4/fengx463/mount_folder/cuda_pcg')
sys.path.insert(0, '/users/4/fengx463/hmc/qed_fermion/qed_fermion')

import torch
from qed_fermion_module import _C
from hmc_sampler_batch import HmcSampler

# HMC inputs
Lx, Ly, Ltau = 2, 2, 2
Vs = Lx * Lx
print(f'max boson idx at tau=0, {Vs*2}')
hmc = HmcSampler(Lx=Lx, Ltau=Ltau)
hmc.dtau = 0.1
hmc.reset()

hmc.initialize_boson_test()

R_u = hmc.draw_psudo_fermion()  # cdtype

#------- batch_size = 1 -------     
psi_u = R_u.to(torch.complex64) # [bs, Ltau*Ly*Lx]
boson = hmc.boson

mat = hmc.get_diag_B_test(boson)
indices = mat.coalesce().indices()
values = mat.coalesce().values()
filtered_indices = indices[:, indices[0] < Lx * Ly]
filtered_values = values[indices[0] < Lx * Ly]
print("mat (COO format): filtered indices=", filtered_indices, "filtered values=", filtered_values)
print("psi_u:", psi_u.view(-1)[:Lx*Ly])

expected = torch.sparse.mm(mat, psi_u.T).to_dense()
expected = expected.T

#
boson = boson.permute(0, 4, 3, 2, 1).to(torch.float32).contiguous()
psi_u = psi_u.view(Ltau, -1)
out = torch.empty_like(psi_u)

for tau in range(Ltau):
    out[tau] = _C.b_vec_per_tau(boson[0, tau].view(-1), psi_u[tau], Lx, 0.1)

# Test if the output is close to the input
torch.testing.assert_close(out.view(-1), expected.view(-1), rtol=1e-5, atol=1e-8)
print("Success!")


# #-------- batch_size > 1 -------
bs = 2
psi_u = R_u.to(torch.complex64).view(1, -1).repeat(bs, 1) # [bs, Ltau*Ly*Lx]
psi_u = psi_u.view(bs, Ltau, -1)

#
boson = hmc.boson.permute(0, 4, 3, 2, 1).reshape(hmc.bs, -1).to(torch.float32).contiguous()
boson = boson.repeat(bs, 1)
boson = boson.view(bs, Ltau, -1)

expected = torch.empty_like(psi_u)  # [bs, Ltau, Lx*Ly]
for b in range(bs):
    B = hmc.get_diag_B_test(hmc.boson[b])
    expected[b] = torch.sparse.mm(B, psi_u[b].view(-1).unsqueeze(1)).to_dense().view(Ltau, -1)

out = torch.empty_like(psi_u)
psi_u = psi_u.view(bs, Ltau, -1)
for i in range(bs): 
    for tau in range(Ltau):
        out[i, tau] = _C.b_vec_per_tau(boson[i, tau], psi_u[i, tau], Lx, 0.1)

# Test if the output is close to the input
torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-8)
print("Success!")
