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
# out = torch.zeros_like(psi_u, dtype=torch.complex64)

out = _C.precon_vec(psi_u, 
                precon,
                Lx)
print('----------python------------')
# print("Result of PCG:", out[-10:], out.shape)

# print out precon and vec info
crow = precon.crow_indices().to(torch.int32)
col = precon.col_indices().to(torch.int32)
val = precon.values().to(torch.complex64)    
# Extract the first row of the sparse matrix `precon`
first_row_start = crow[0].item()
first_row_end = crow[1].item()
first_row_cols = col[first_row_start:first_row_end]
first_row_vals = val[first_row_start:first_row_end]

# print("First row columns:", first_row_cols.tolist())
# print("First row values:", first_row_vals.tolist())

# # Print the first row of `precon`
# print("First row of precon:")
# for i, (col_idx, value) in enumerate(zip(first_row_cols, first_row_vals)):
#     print(f"Entry {i}: Column {col_idx.item()}, Value {value.item()}")
#     print(f"psi_u[{col_idx.item()}]: {psi_u[col_idx.item()].item()}")

# Extract the last row of the sparse matrix `precon`
last_row_start = crow[-1*Vs-1].item()
last_row_end = crow[-1*Vs].item()
last_row_cols = col[last_row_start:last_row_end]
last_row_vals = val[last_row_start:last_row_end]

print("Last row columns:", last_row_cols.tolist())
print("Last row values:", last_row_vals.tolist())

# Print the last row of `precon`
print("Last row of precon:")
for i, (col_idx, value) in enumerate(zip(last_row_cols, last_row_vals)):
    print(f"Entry {i}: Column {col_idx.item()}, Value {value.item()}")
    print(f"psi_u[{col_idx.item()}]: {psi_u[col_idx.item()].item()}")

expected = torch.sparse.mm(precon, psi_u).to_dense()
# print("Expected result:", expected[-10:], expected.shape)
# torch.testing.assert_close(expected, out, atol=1e-6, rtol=0)




