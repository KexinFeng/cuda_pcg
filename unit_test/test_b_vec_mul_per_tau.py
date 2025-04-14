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
boson = hmc.boson

mat = hmc.get_diag_B_test(boson)

expected = torch.sparse.mm(mat, psi_u.T).to_dense()
expected = expected.T

#
boson = boson.permute(0, 4, 3, 2, 1).to(torch.float32).contiguous()
psi_u = psi_u.view(Ltau, -1)
out = torch.empty_like(psi_u)

for tau in range(Ltau):
    out[tau] = _C.b_vec_per_tau(boson[0, tau].view(-1), psi_u[tau], Lx, 0.1, False)

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
    B = hmc.get_diag_B_test(hmc.boson)
    expected[b] = torch.sparse.mm(B, psi_u[b].view(-1).unsqueeze(1)).to_dense().view(Ltau, -1)

out = torch.empty_like(psi_u)
psi_u = psi_u.view(bs, Ltau, -1)
for i in range(bs): 
    for tau in range(Ltau):
        out[i, tau] = _C.b_vec_per_tau(boson[i, tau], psi_u[i, tau], Lx, 0.1, False)

# Test if the output is close to the input
torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-8)
print("Success!")

# #-------- batch_size = 1 interm_out -------
psi_u = R_u.to(torch.complex64) # [bs, Ltau*Ly*Lx]
boson = hmc.boson

results = hmc.get_M_sparse(boson)
B_list = results[1]
B1_list = B_list[0]
B2_list = B_list[1]
B3_list = B_list[2]
B4_list = B_list[3]

#
boson = boson.permute(0, 4, 3, 2, 1).to(torch.float32).contiguous()
psi_u = psi_u.view(Ltau, -1)

out = torch.empty_like(psi_u).unsqueeze(0)
out = out.repeat(6, 1, 1)

for tau in range(Ltau):
    interm_out = _C.b_vec_per_tau(boson[0, tau].view(-1), psi_u[tau], Lx, 0.1, True)
    out[:, tau, :] = interm_out.view(6, -1)

    xi_n_lft_5 = psi_u[tau].conj().view(1, -1) # row
    xi_n_lft_4 = torch.sparse.mm(xi_n_lft_5, B4_list[tau])
    xi_n_lft_3 = torch.sparse.mm(xi_n_lft_4, B3_list[tau])
    xi_n_lft_2 = torch.sparse.mm(xi_n_lft_3, B2_list[tau])
    xi_n_lft_1 = torch.sparse.mm(xi_n_lft_2, B1_list[tau])
    xi_n_lft_0 = torch.sparse.mm(xi_n_lft_1, B2_list[tau])
    xi_n_lft_m1 = torch.sparse.mm(xi_n_lft_0, B3_list[tau])

    # Test if the output is close to the input
    torch.testing.assert_close(out[0, tau].view(-1).conj(), xi_n_lft_4.view(-1), rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(out[1, tau].view(-1).conj(), xi_n_lft_3.view(-1), rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(out[2, tau].view(-1).conj(), xi_n_lft_2.view(-1), rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(out[3, tau].view(-1).conj(), xi_n_lft_1.view(-1), rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(out[4, tau].view(-1).conj(), xi_n_lft_0.view(-1), rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(out[5, tau].view(-1).conj(), xi_n_lft_m1.view(-1), rtol=1e-5, atol=1e-8)

print("Success!")

# #-------- batch_size > 1 interm_out -------
bs = 2

boson = hmc.boson

results = hmc.get_M_sparse(boson)
B_list = results[1]
B1_list = B_list[0]
B2_list = B_list[1]
B3_list = B_list[2]
B4_list = B_list[3]

psi_u = R_u.to(torch.complex64).view(1, -1).repeat(bs, 1) # [bs, Ltau*Ly*Lx]
psi_u = psi_u.view(bs, Ltau, -1)

out = torch.empty_like(psi_u).unsqueeze(0)
out = out.repeat(6, 1, 1, 1)
boson = boson.permute(0, 4, 3, 2, 1).to(torch.float32).contiguous()
for b in range(bs):
    for tau in range(Ltau):
        interm_out = _C.b_vec_per_tau(boson[0, tau].view(-1), psi_u[b, tau], Lx, 0.1, True)
        out[:, b, tau, :] = interm_out.view(6, -1)

        xi_n_lft_5 = psi_u[b, tau].conj().view(1, -1) # row
        xi_n_lft_4 = torch.sparse.mm(xi_n_lft_5, B4_list[tau])
        xi_n_lft_3 = torch.sparse.mm(xi_n_lft_4, B3_list[tau])
        xi_n_lft_2 = torch.sparse.mm(xi_n_lft_3, B2_list[tau])
        xi_n_lft_1 = torch.sparse.mm(xi_n_lft_2, B1_list[tau])
        xi_n_lft_0 = torch.sparse.mm(xi_n_lft_1, B2_list[tau])
        xi_n_lft_m1 = torch.sparse.mm(xi_n_lft_0, B3_list[tau])

        # Test if the output is close to the input
        torch.testing.assert_close(out[0, b, tau].view(-1).conj(), xi_n_lft_4.view(-1), rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(out[1, b, tau].view(-1).conj(), xi_n_lft_3.view(-1), rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(out[2, b, tau].view(-1).conj(), xi_n_lft_2.view(-1), rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(out[3, b, tau].view(-1).conj(), xi_n_lft_1.view(-1), rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(out[4, b, tau].view(-1).conj(), xi_n_lft_0.view(-1), rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(out[5, b, tau].view(-1).conj(), xi_n_lft_m1.view(-1), rtol=1e-5, atol=1e-8)

print("Success!")


