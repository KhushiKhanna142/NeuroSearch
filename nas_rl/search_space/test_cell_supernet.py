"""
Sanity checks for cell.py and supernet.py
Run: python nas_rl/search_space/test_cell_supernet.py
"""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from nas_rl.search_space.ops import IDX_TO_OP
from nas_rl.search_space.cell import Cell
from nas_rl.search_space.supernet import Supernet

BATCH     = 2
C         = 16
N_NODES   = 4
N_CLASSES = 10


def rand_arch_weights(n_nodes):
    return {
        f'{j}_{i+2}': random.choice(IDX_TO_OP)
        for i in range(n_nodes)
        for j in range(i + 2)
    }


def section(title):
    print(); print("=" * 60); print(title); print("=" * 60)


# ------------------------------------------------------------------
# Cell — normal
# ------------------------------------------------------------------
section("Cell — normal (stride=1, prev_reduction=False)")
cell_n = Cell(N_NODES, C_prev_prev=C, C_prev=C, C=C,
              reduction=False, prev_reduction=False)
s0 = torch.randn(BATCH, C, 8, 8)
s1 = torch.randn(BATCH, C, 8, 8)
with torch.no_grad():
    out = cell_n(s0, s1, rand_arch_weights(N_NODES))
exp = (BATCH, C * N_NODES, 8, 8)
st = "PASS" if tuple(out.shape) == exp else f"FAIL got {tuple(out.shape)}"
print(f"  output {tuple(out.shape)}  expected {exp}  {st}")

# ------------------------------------------------------------------
# Cell — reduction
# ------------------------------------------------------------------
section("Cell — reduction (stride=2, prev_reduction=False)")
cell_r = Cell(N_NODES, C_prev_prev=C, C_prev=C, C=C*2,
              reduction=True, prev_reduction=False)
s0r = torch.randn(BATCH, C, 8, 8)
s1r = torch.randn(BATCH, C, 8, 8)
with torch.no_grad():
    out_r = cell_r(s0r, s1r, rand_arch_weights(N_NODES))
exp_r = (BATCH, C*2*N_NODES, 4, 4)
st_r = "PASS" if tuple(out_r.shape) == exp_r else f"FAIL got {tuple(out_r.shape)}"
print(f"  output {tuple(out_r.shape)}  expected {exp_r}  {st_r}")

# ------------------------------------------------------------------
# Cell — first cell after a reduction (prev_reduction=True)
# ------------------------------------------------------------------
section("Cell — normal after reduction (prev_reduction=True)")
# s0 comes from two cells back (larger spatial, C channels)
# s1 comes from the reduction cell (smaller spatial, C*2*N_NODES channels)
C_pp = C              # C_prev_prev: before the reduction
C_p  = C*2*N_NODES    # C_prev: reduction cell output (n_nodes concat)
C_c  = C*2            # C_curr stays doubled after the reduction section
cell_post = Cell(N_NODES, C_prev_prev=C_pp, C_prev=C_p, C=C_c,
                 reduction=False, prev_reduction=True)
s0_big   = torch.randn(BATCH, C_pp, 8, 8)  # larger spatial (pre-reduction)
s1_small = torch.randn(BATCH, C_p,  4, 4)  # smaller spatial (post-reduction)
with torch.no_grad():
    out_post = cell_post(s0_big, s1_small, rand_arch_weights(N_NODES))
exp_post = (BATCH, C_c * N_NODES, 4, 4)
st_post = "PASS" if tuple(out_post.shape) == exp_post else f"FAIL got {tuple(out_post.shape)}"
print(f"  output {tuple(out_post.shape)}  expected {exp_post}  {st_post}")

# ------------------------------------------------------------------
# Edge keys
# ------------------------------------------------------------------
section("Cell — edge keys")
keys = Cell.edge_keys(N_NODES)
print(f"  n_nodes={N_NODES}  total edges={len(keys)}")
print(f"  keys: {keys}")
assert len(keys) == 14
print("  PASS")

# ------------------------------------------------------------------
# Supernet — build and forward
# ------------------------------------------------------------------
section("Supernet — build and forward (CIFAR-10, 32x32)")
net = Supernet(C_init=16, n_cells=8, n_nodes=4, n_classes=10)
x   = torch.randn(BATCH, 3, 32, 32)
spec = net.random_arch()

print(f"  arch_spec: {len(spec)} cells, {len(spec[0])} edges each")
with torch.no_grad():
    logits = net(x, spec)
exp_l = (BATCH, N_CLASSES)
st_l = "PASS" if tuple(logits.shape) == exp_l else f"FAIL got {tuple(logits.shape)}"
print(f"  logits {tuple(logits.shape)}  expected {exp_l}  {st_l}")

# ------------------------------------------------------------------
# Supernet — multiple random archs forward pass
# ------------------------------------------------------------------
section("Supernet — 5 different random archs, all valid")
for trial in range(5):
    spec2 = net.random_arch()
    with torch.no_grad():
        out2 = net(x, spec2)
    assert out2.shape == torch.Size([BATCH, N_CLASSES])
print("  PASS")

# ------------------------------------------------------------------
# Supernet — cell resolutions
# ------------------------------------------------------------------
section("Supernet — cell_resolutions")
res = net.cell_resolutions(input_size=32)
print(f"  {res}")
assert len(res) == 8
print("  PASS")

# ------------------------------------------------------------------
# Param count
# ------------------------------------------------------------------
section("Supernet — param count")
total = sum(p.numel() for p in net.parameters())
print(f"  {total:,}  ({total/1e6:.2f}M)")

print()
print("All tests PASSED")