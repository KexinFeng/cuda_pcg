import sys

sys.path.insert(0, '~/mount_folder/cuda_pcg')
sys.path.insert(0, '~/hmc/qed_fermion/qed_fermion')

import torch
import qed_fermion_module._C as core
from hmc_sampler_batch import HmcSampler


a = torch.randn(3, 3, device='cuda')
b = torch.randn(3, 3, device='cuda')

print("Tensor A:", a)
print("Tensor B:", b)

result = core.add_tensors(a, b)
print("Result (A + B):", result)

# HMC inputs
Lx, Ly, Ltau = 2, 2, 40
hmc = HmcSampler(Lx=Lx, Ltau=Ltau)

boson = hmc.boson.to(torch.float32)  # dtype
# hmc.i_list_1  # dtype
# hmc.i_list_2
# hmc.i_list_3
# hmc.i_list_4

R_u = hmc.draw_psudo_fermion().view(-1, 1)  # cdtype
# result = hmc.get_M_sparse(hmc.boson)
# MhM0, B_list, M0 = result[0], result[1], result[-1]
# psi_u = torch.sparse.mm(M0.permute(1, 0).conj(), R_u)

psi_u = R_u.to(torch.complex64)

hmc.reset_precon()
precon = hmc.precon.to(torch.complex64)
# precon_ind = precon.indices()
# precon_val = precon.values()
# precon_sz = precon.size()
i_lists = torch.stack([hmc.i_list_1, 
                       hmc.i_list_2, 
                       hmc.i_list_3, 
                       hmc.i_list_4])
j_lists = torch.stack([hmc.j_list_1, 
                       hmc.j_list_2, 
                       hmc.j_list_3, 
                       hmc.j_list_4])

max_iter = 50
result = core.pcg(i_lists, 
        j_lists,
        boson, 
        psi_u, 
        precon, 
        Lx, Ltau, max_iter, 1e-6
)
print("Result of PCG:", result[:10], result.shape)

# Expected output
expected = torch.tensor([
        [-1.0282+0.7633j],
        [-0.5651+0.3180j],
        [-2.0545+0.4874j],
        [ 0.8165-0.3189j],
        [-0.8099+1.4335j],
        [-0.6698+1.1058j],
        [-1.2660-0.3112j],
        [ 1.2133-0.8051j],
        [-1.0215+1.4854j],
        [-0.6699+1.5203j]], device='cuda', dtype=torch.complex64)

torch.testing.assert_close(result[:10], expected, rtol=1e-4, atol=1e-6)

