import sys

sys.path.insert(0, '/users/4/fengx463/mount_folder/awsome_project')
sys.path.insert(0, '/users/4/fengx463/hmc/qed_fermion/qed_fermion')

import torch
import qed_fermion
from hmc_sampler_batch import HmcSampler


a = torch.randn(3, 3, device='cuda')
b = torch.randn(3, 3, device='cuda')

print("Tensor A:", a)
print("Tensor B:", b)

result = qed_fermion._C.add_tensors(a, b)
print("Result (A + B):", result)

# HMC inputs
Lx, Ly, Ltau = 2, 2, 40
hmc = HmcSampler(Lx=Lx, Ltau=Ltau)

boson = hmc.boson  # dtype
# hmc.i_list_1  # dtype
# hmc.i_list_2
# hmc.i_list_3
# hmc.i_list_4

R_u = hmc.draw_psudo_fermion().view(-1, 1)  # cdtype
# result = hmc.get_M_sparse(hmc.boson)
# MhM0, B_list, M0 = result[0], result[1], result[-1]
# psi_u = torch.sparse.mm(M0.permute(1, 0).conj(), R_u)

psi_u = R_u

precon = hmc.precon
# precon_ind = precon.indices()
# precon_val = precon.values()
# precon_sz = precon.size()
i_lists = [hmc.i_list_1, hmc.i_list_2, hmc.i_list_3, hmc.i_list_4]
j_lists = [hmc.j_list_1, hmc.j_list_2, hmc.j_list_3, hmc.j_list_4]

max_iter = 50
qed_fermion._C.pcg(i_lists, 
                   j_lists,
                   boson, 
                   psi_u, 
                   precon, 
                   Lx, Ltau, max_iter, 1e-6
)



