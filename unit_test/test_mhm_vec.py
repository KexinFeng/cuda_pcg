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
hmc.initialize_boson_test()

R_u = hmc.draw_psudo_fermion()  # cdtype

#------- batch_size = 1 -------     
psi_u = R_u.to(torch.complex64)

boson = hmc.boson.permute(0, 4, 3, 2, 1).reshape(hmc.bs, -1).to(torch.float32).contiguous()
# boson = torch.arange(boson.numel(), device=boson.device).reshape(boson.shape).to(torch.float32)

psi_u = torch.arange(psi_u.numel(), device=psi_u.device).reshape(psi_u.shape).to(psi_u.dtype)

out = _C.mhm_vec(boson, psi_u, Lx, float(1e-8))
print("Result:", out[0, :10], out.shape)

print("Input:", psi_u[0, :10], psi_u.shape)

# Test if the output is close to the input
torch.testing.assert_close(out, psi_u, rtol=1e-5, atol=1e-8)
print("Success!")


