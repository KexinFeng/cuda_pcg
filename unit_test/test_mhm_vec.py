import sys

sys.path.insert(0, '/users/4/fengx463/mount_folder/cuda_pcg')
sys.path.insert(0, '/users/4/fengx463/hmc/qed_fermion/qed_fermion')

import torch
from qed_fermion_module import _C
from hmc_sampler_batch import HmcSampler

# HMC inputs
Lx, Ly, Ltau = 10, 10, 400
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
# indices = mat.coalesce().indices()
# values = mat.coalesce().values()
# filtered_indices = indices[:, indices[0] < Lx * Ly]
# filtered_values = values[indices[0] < Lx * Ly]
# print("mat (COO format): filtered indices=", filtered_indices, "filtered values=", filtered_values)
# print("psi_u:", psi_u.view(-1)[:Lx*Ly])

#         return MhM, [B1_list, B2_list, B3_list, B4_list], blk_sparsity, M
mhm, _, _, M = hmc.get_M_sparse(hmc.boson)
expected = torch.sparse.mm(mhm, psi_u.T).to_dense()
expected = expected.T
print('expected:', expected)

interm_vec = torch.sparse.mm(M, psi_u.T).to_dense()
interm_vec = interm_vec.T
print('interm_vec:', interm_vec)

#
boson = hmc.boson.permute(0, 4, 3, 2, 1).reshape(hmc.bs, -1).to(torch.float32).contiguous()
out = _C.mhm_vec(boson, psi_u, Lx, 0.1)
print('------')
print(out.shape)
print(out)

# Test if the output is close to the input
torch.testing.assert_close(out[:1], expected, rtol=2e-5, atol=1e-7)
# print("Success!")


# # #-------- batch_size > 1 -------
bs = 2
psi_u = R_u.to(torch.complex64).view(1, -1).repeat(bs, 1) # [bs, Ltau*Ly*Lx]
mhm, _, _, M = hmc.get_M_sparse(hmc.boson)
expected = torch.empty_like(psi_u)
for i in range(bs):
    expected[i] = torch.sparse.mm(mhm, psi_u[i:i+1].T).to_dense().view(-1)

boson = hmc.boson.permute(0, 4, 3, 2, 1).reshape(hmc.bs, -1).to(torch.float32).contiguous()
boson = boson.repeat(bs, 1)
out = _C.mhm_vec(boson, psi_u, Lx, 0.1)

# Test if the output is close to the input
torch.testing.assert_close(out[:bs], expected, rtol=2e-5, atol=1e-7)
print("Success!")
