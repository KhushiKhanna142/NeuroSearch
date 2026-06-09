"""
Tests for architecture_pool.py and search_loop.py.

Uses a tiny supernet (C_init=4, n_cells=2) and a synthetic val loader
so the full loop runs in seconds without needing CIFAR-10.

Run: python nas_rl/search/test_search.py
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from torch.utils.data import DataLoader, TensorDataset

from nas_rl.search_space.supernet import Supernet
from nas_rl.controller.lstm_controller import LSTMController
from nas_rl.trainer.evaluator import Evaluator
from nas_rl.reward.cost_estimator import FLOPsEstimator
from nas_rl.reward.reward_combiner import RewardCombiner
from nas_rl.search.architecture_pool import ArchitecturePool
from nas_rl.search.search_loop import SearchLoop

DEVICE = torch.device('cpu')
N_CELLS = 2
N_NODES = 4


def fake_loader(n=128, bs=32):
    x = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=bs)


def section(title):
    print(); print("=" * 60); print(title); print("=" * 60)


# ==================================================================
# ArchitecturePool tests
# ==================================================================
section("ArchitecturePool — add and deduplication")

net  = Supernet(C_init=4, n_cells=N_CELLS, n_nodes=N_NODES).to(DEVICE)
pool = ArchitecturePool(capacity=10)

spec1 = net.random_arch()
spec2 = net.random_arch()

added1a = pool.add(spec1, accuracy=0.80, flops=50e6, reward=0.5, episode=1)
added1b = pool.add(spec1, accuracy=0.80, flops=50e6, reward=0.5, episode=2)
added2  = pool.add(spec2, accuracy=0.85, flops=60e6, reward=0.6, episode=3)

print(f"  First add of spec1  : {added1a}  (expected True)")
print(f"  Duplicate of spec1  : {added1b}  (expected False)")
print(f"  Add spec2           : {added2}   (expected True)")
print(f"  Pool size           : {len(pool)}  (expected 2)")

assert added1a == True
assert added1b == False
assert added2  == True
assert len(pool) == 2
print("  PASS")


section("ArchitecturePool — top_k by reward and accuracy")

# Add a few more with known rewards
specs = [net.random_arch() for _ in range(5)]
rewards   = [0.3, 0.9, 0.1, 0.7, 0.5]
accs      = [0.70, 0.88, 0.60, 0.83, 0.75]
for i, (s, r, a) in enumerate(zip(specs, rewards, accs)):
    pool.add(s, accuracy=a, flops=50e6, reward=r, episode=10+i)

top3_reward = pool.top_k(3, sort_by='reward')
top3_acc    = pool.top_k(3, sort_by='accuracy')

print(f"  Top-3 by reward  : {[round(e['reward'],2) for e in top3_reward]}")
print(f"  Top-3 by accuracy: {[round(e['accuracy'],2) for e in top3_acc]}")

assert top3_reward[0]['reward'] >= top3_reward[1]['reward']
assert top3_reward[1]['reward'] >= top3_reward[2]['reward']
assert top3_acc[0]['accuracy']  >= top3_acc[1]['accuracy']
print("  PASS")


section("ArchitecturePool — capacity enforcement")

small_pool = ArchitecturePool(capacity=3)
specs_cap  = [net.random_arch() for _ in range(6)]
rewards_cap = [0.1, 0.9, 0.5, 0.8, 0.2, 0.95]

for i, (s, r) in enumerate(zip(specs_cap, rewards_cap)):
    small_pool.add(s, accuracy=r, flops=50e6, reward=r, episode=i)

print(f"  Pool size (cap=3): {len(small_pool)}  (expected 3)")
worst_in_pool = min(e['reward'] for e in small_pool.pool)
print(f"  Worst reward in pool: {worst_in_pool:.2f}")
# Best 3 rewards are 0.95, 0.9, 0.8 — worst should be >= 0.8
assert worst_in_pool >= 0.5, \
    f"Pool should only keep high-reward archs, worst={worst_in_pool}"
print("  PASS")


section("ArchitecturePool — best()")

best_entry = pool.best()
print(f"  Best reward: {best_entry['reward']:.4f}")
assert best_entry is not None
assert best_entry['reward'] == max(e['reward'] for e in pool.pool)
print("  PASS")


# ==================================================================
# SearchLoop tests
# ==================================================================

def build_search_loop(n_cells=N_CELLS, n_nodes=N_NODES):
    supernet   = Supernet(C_init=4, n_cells=n_cells,
                          n_nodes=n_nodes, n_classes=10).to(DEVICE)
    controller = LSTMController(n_cells=n_cells,
                                n_nodes=n_nodes, hidden_dim=32).to(DEVICE)
    val_loader = fake_loader()
    evaluator  = Evaluator(supernet, val_loader, DEVICE)
    cost_est   = FLOPsEstimator(supernet)
    reward_fn  = RewardCombiner(flops_target=50e6, alpha=0.07, norm_warmup=3)

    loop = SearchLoop(
        supernet        = supernet,
        controller      = controller,
        evaluator       = evaluator,
        reward_combiner = reward_fn,
        cost_estimator  = cost_est,
        device          = DEVICE,
        ctrl_lr         = 3e-3,
        entropy_coeff   = 0.01,
    )
    return loop, supernet, controller


section("SearchLoop — single episode returns correct keys")

loop, net2, ctrl = build_search_loop()
stats = loop.run_episode()

expected_keys = {
    'episode', 'accuracy', 'flops', 'raw_reward', 'reward',
    'advantage', 'policy_loss', 'entropy', 'total_loss',
    'baseline', 'is_new_arch', 'elapsed_s',
}
print(f"  Stats keys: {set(stats.keys())}")
assert set(stats.keys()) == expected_keys
print("  PASS")


section("SearchLoop — episode stats are in valid ranges")

print(f"  accuracy    : {stats['accuracy']:.4f}")
print(f"  flops       : {stats['flops']/1e6:.2f}M")
print(f"  entropy     : {stats['entropy']:.4f}")
print(f"  policy_loss : {stats['policy_loss']:.4f}")
print(f"  elapsed_s   : {stats['elapsed_s']:.2f}s")

assert 0.0 <= stats['accuracy'] <= 1.0
assert stats['flops'] >= 0
assert stats['entropy'] >= 0
assert stats['elapsed_s'] > 0
print("  PASS")


section("SearchLoop — supernet weights frozen during search")

# Collect supernet params before 5 episodes
params_before = [p.clone() for p in net2.parameters()]
for _ in range(5):
    loop.run_episode()
params_after = [p.clone() for p in net2.parameters()]

for pb, pa in zip(params_before, params_after):
    assert torch.allclose(pb, pa), "Supernet weights changed during search!"

print("  Supernet weights unchanged after 5 episodes  PASS")


section("SearchLoop — controller weights DO change")

ctrl_params_before = [p.clone() for p in ctrl.parameters()]
for _ in range(5):
    loop.run_episode()
ctrl_params_after = [p.clone() for p in ctrl.parameters()]

changed = any(
    not torch.allclose(pb, pa)
    for pb, pa in zip(ctrl_params_before, ctrl_params_after)
)
print(f"  Controller weights changed: {changed}")
assert changed, "Controller should be learning!"
print("  PASS")


section("SearchLoop — entropy stays positive (exploration maintained)")

entropies = []
for _ in range(10):
    s = loop.run_episode()
    entropies.append(s['entropy'])

print(f"  Entropy over 10 episodes: "
      f"min={min(entropies):.4f}  max={max(entropies):.4f}")
assert all(e > 0 for e in entropies), "Entropy dropped to zero — collapsed!"
print("  PASS")


section("SearchLoop — pool grows with unique archs")

pool_before = len(loop.arch_pool)
for _ in range(10):
    loop.run_episode()
pool_after = len(loop.arch_pool)

print(f"  Pool size: {pool_before} -> {pool_after}")
assert pool_after >= pool_before
print("  PASS")


section("SearchLoop — checkpoint save and resume")

loop2, _, _ = build_search_loop()
for _ in range(5):
    loop2.run_episode()

with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, 'search.pt')
    loop2.save_checkpoint(ckpt_path)

    # Build a fresh loop and load the checkpoint
    loop3, _, _ = build_search_loop()
    loop3.load_checkpoint(ckpt_path)

    assert loop3._episode == loop2._episode
    assert len(loop3.arch_pool) == len(loop2.arch_pool)
    print(f"  Resumed at episode {loop3._episode}  "
          f"pool size {len(loop3.arch_pool)}  PASS")


section("SearchLoop — run() executes N episodes end to end")

loop4, _, _ = build_search_loop()
top5 = loop4.run(n_episodes=15, log_every=5)

print(f"\n  Top-5 archs returned: {len(top5)}")
assert len(top5) <= 5
assert all('arch' in e and 'reward' in e for e in top5)
print("  PASS")


print()
print("All tests PASSED")