"""
Sanity checks for lstm_controller.py and baseline.py
Run: python nas_rl/controller/test_controller.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from nas_rl.search_space.ops import IDX_TO_OP, N_OPS
from nas_rl.search_space.cell import Cell
from nas_rl.controller.lstm_controller import LSTMController
from nas_rl.controller.baseline import ExponentialBaseline

N_CELLS  = 8
N_NODES  = 4
N_EDGES  = len(Cell.edge_keys(N_NODES))   # 14
N_DEC    = N_CELLS * N_EDGES              # 112
DEVICE   = torch.device('cpu')


def section(title):
    print(); print("=" * 60); print(title); print("=" * 60)


# ------------------------------------------------------------------
# Controller — build
# ------------------------------------------------------------------
section("LSTMController — instantiate")
ctrl = LSTMController(n_cells=N_CELLS, n_nodes=N_NODES, hidden_dim=64)
total_params = sum(p.numel() for p in ctrl.parameters())
print(f"  Parameters: {total_params:,}")
print(f"  n_decisions expected: {N_DEC}")
assert ctrl.n_decisions == N_DEC
print("  PASS")


# ------------------------------------------------------------------
# Controller — forward shapes
# ------------------------------------------------------------------
section("LSTMController — forward() output shapes")
arch_spec, log_probs, entropies = ctrl(DEVICE)

print(f"  arch_spec length       : {len(arch_spec)}  (expected {N_CELLS})")
print(f"  arch_spec[0] n_edges   : {len(arch_spec[0])}  (expected {N_EDGES})")
print(f"  log_probs shape        : {tuple(log_probs.shape)}  (expected ({N_DEC},))")
print(f"  entropies shape        : {tuple(entropies.shape)}  (expected ({N_DEC},))")

assert len(arch_spec) == N_CELLS
assert len(arch_spec[0]) == N_EDGES
assert log_probs.shape  == torch.Size([N_DEC])
assert entropies.shape  == torch.Size([N_DEC])
print("  PASS")


# ------------------------------------------------------------------
# Controller — arch_spec validity
# ------------------------------------------------------------------
section("LSTMController — arch_spec contains valid op names")
edge_keys = Cell.edge_keys(N_NODES)
for cell_idx, cell_dict in enumerate(arch_spec):
    for key in edge_keys:
        assert key in cell_dict, f"Missing key {key} in cell {cell_idx}"
        assert cell_dict[key] in IDX_TO_OP, \
            f"Unknown op '{cell_dict[key]}' at cell {cell_idx} edge {key}"
print(f"  All {N_CELLS * N_EDGES} edge decisions are valid op names  PASS")


# ------------------------------------------------------------------
# Controller — log_probs are negative (log of probability <= 1)
# ------------------------------------------------------------------
section("LSTMController — log_probs <= 0")
assert (log_probs <= 0).all(), "Some log_probs are positive — bug in sampling"
print(f"  min={log_probs.min():.4f}  max={log_probs.max():.4f}  PASS")


# ------------------------------------------------------------------
# Controller — entropies are non-negative and <= log(N_OPS)
# ------------------------------------------------------------------
section("LSTMController — entropies in valid range")
import math
max_entropy = math.log(N_OPS)
assert (entropies >= 0).all()
assert (entropies <= max_entropy + 1e-4).all()
print(f"  range [{entropies.min():.4f}, {entropies.max():.4f}]  "
      f"max possible {max_entropy:.4f}  PASS")


# ------------------------------------------------------------------
# Controller — stochasticity: two samples differ
# ------------------------------------------------------------------
section("LSTMController — stochastic sampling produces different archs")
specs = [ctrl(DEVICE)[0] for _ in range(10)]
op_sets = [
    tuple(cell[k] for cell in s for k in edge_keys)
    for s in specs
]
n_unique = len(set(op_sets))
print(f"  Unique archs in 10 samples: {n_unique}")
assert n_unique > 1, "Controller is deterministic — something is wrong"
print("  PASS")


# ------------------------------------------------------------------
# Controller — greedy_arch is deterministic
# ------------------------------------------------------------------
section("LSTMController — greedy_arch() is deterministic")
g1 = ctrl.greedy_arch(DEVICE)
g2 = ctrl.greedy_arch(DEVICE)
g1_flat = tuple(cell[k] for cell in g1 for k in edge_keys)
g2_flat = tuple(cell[k] for cell in g2 for k in edge_keys)
assert g1_flat == g2_flat, "greedy_arch returned different results — bug"
print(f"  Same arch on two calls  PASS")


# ------------------------------------------------------------------
# Controller — backward pass works (needed for REINFORCE)
# ------------------------------------------------------------------
section("LSTMController — backward pass through log_probs")
arch_spec2, log_probs2, entropies2 = ctrl(DEVICE)
fake_reward   = torch.tensor(0.5)
fake_baseline = torch.tensor(0.3)
loss = -(log_probs2 * (fake_reward - fake_baseline)).sum()
loss -= 0.01 * entropies2.sum()
loss.backward()
grad_norms = [p.grad.norm().item() for p in ctrl.parameters()
              if p.grad is not None]
print(f"  Loss: {loss.item():.4f}")
print(f"  Params with gradients: {len(grad_norms)}")
print(f"  Grad norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
assert len(grad_norms) > 0
print("  PASS")


# ------------------------------------------------------------------
# Baseline — EMA behaviour
# ------------------------------------------------------------------
section("ExponentialBaseline — cold start and update")
bl = ExponentialBaseline(decay=0.95)
assert bl.get() == 0.0, "Should return 0.0 before any rewards"

rewards = [0.5, 0.6, 0.4, 0.8, 0.7]
values  = [bl.update(r) for r in rewards]
print(f"  Rewards : {rewards}")
print(f"  Baseline: {[f'{v:.4f}' for v in values]}")

# First update should equal the first reward exactly
assert values[0] == rewards[0], "Cold start should set baseline = first reward"

# Baseline should track rewards (roughly monotone upward here)
assert values[-1] > values[0], "Baseline should drift toward higher rewards"
print("  PASS")


section("ExponentialBaseline — reset")
bl.reset()
assert bl.get() == 0.0
print("  PASS")


print()
print("All tests PASSED")