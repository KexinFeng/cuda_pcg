import sys

sys.path.insert(0, '/users/4/fengx463/mount_folder/cuda_pcg')
sys.path.insert(0, '/users/4/fengx463/hmc/qed_fermion/qed_fermion')

import torch
from qed_fermion_module import _C
from hmc_sampler_batch import HmcSampler

# HMC inputs
Lx, Ly, Ltau = 2, 2, 80
Vs = Lx * Lx
hmc = HmcSampler(Lx=Lx, Ltau=Ltau)
hmc.initialize_boson_test()

R_u = hmc.draw_psudo_fermion()  # cdtype

#------- batch_size = 1 -------     
psi_u = R_u.to(torch.complex64)

boson = hmc.boson.permute(0, 4, 3, 2, 1).reshape(hmc.bs, -1).to(torch.float32)

out = _C.mhm_vec(boson, psi_u, Lx, float(0.1)).view(-1, 1)
print("Result:", out[-10:], out.shape)

out_vec = torch.sparse.mm(boson, psi_u)
print(out_vec[-10:], out_vec.shape)


