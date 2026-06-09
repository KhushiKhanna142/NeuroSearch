"""
Tests for cost_estimator.py and reward_combiner.py.
Run: python nas_rl/reward/test_reward.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
from nas_rl.search_space.supernet import Supernet
from nas_rl.search_space.ops import IDX_TO_OP
from nas_rl.reward.cost_estimator import FLOPsEstimator
from nas_rl.reward.reward_combiner import RewardCombiner


def section(title):
    print(); print("=" * 60); print(title); print("=" * 60)


# ------------------------------------------------------------------
# Build supernet and estimator once
# ------------------------------------------------------------------
net       = Supernet(C_init=16, n_cells=8, n_nodes=4, n_classes=10)
estimator = FLOPsEstimator(net, input_size=32)


# ------------------------------------------------------------------
# FLOPs table is built for every cell
# ------------------------------------------------------------------
section("FLOPsEstimator — lookup table coverage")

assert len(estimator.table) == 8, "Should have one entry per cell"
for cell_idx, cell_table in estimator.table.items():
    assert set(cell_table.keys()) == set(IDX_TO_OP), \
        f"Cell {cell_idx} missing ops"
    print(f"  Cell {cell_idx}: {len(cell_table)} ops — "
          f"sample FLOPs conv3x3={cell_table['conv3x3']:,}")

print("  PASS")


# ------------------------------------------------------------------
# FLOPs estimate is positive and finite
# ------------------------------------------------------------------
section("FLOPsEstimator — estimate() returns positive finite value")

arch_spec = net.random_arch()
flops     = estimator.estimate(arch_spec)
print(f"  Random arch FLOPs: {flops:,}  ({flops/1e6:.2f}M)")
assert flops > 0
assert math.isfinite(flops)
print("  PASS")


# ------------------------------------------------------------------
# zero-op arch has zero FLOPs
# ------------------------------------------------------------------
section("FLOPsEstimator — all-zero arch has 0 FLOPs")

from nas_rl.search_space.cell import Cell
zero_spec = [
    {key: 'zero' for key in Cell.edge_keys(4)}
    for _ in range(8)
]
zero_flops = estimator.estimate(zero_spec)
print(f"  All-zero arch FLOPs: {zero_flops}")
assert zero_flops == 0
print("  PASS")


# ------------------------------------------------------------------
# skip-op arch also has 0 FLOPs
# ------------------------------------------------------------------
section("FLOPsEstimator — all-skip arch has 0 FLOPs")

skip_spec = [
    {key: 'skip' for key in Cell.edge_keys(4)}
    for _ in range(8)
]
skip_flops = estimator.estimate(skip_spec)
print(f"  All-skip arch FLOPs: {skip_flops}")
assert skip_flops == 0
print("  PASS")


# ------------------------------------------------------------------
# Heavy arch (conv3x3 everywhere) > light arch (conv1x1 everywhere)
# ------------------------------------------------------------------
section("FLOPsEstimator — conv3x3 > conv1x1 in FLOPs")

heavy_spec = [
    {key: 'conv3x3' for key in Cell.edge_keys(4)}
    for _ in range(8)
]
light_spec = [
    {key: 'conv1x1' for key in Cell.edge_keys(4)}
    for _ in range(8)
]
heavy_flops = estimator.estimate(heavy_spec)
light_flops = estimator.estimate(light_spec)
print(f"  conv3x3 everywhere : {heavy_flops/1e6:.2f}M FLOPs")
print(f"  conv1x1 everywhere : {light_flops/1e6:.2f}M FLOPs")
assert heavy_flops > light_flops, \
    "conv3x3 should cost more FLOPs than conv1x1"
print("  PASS")


# ------------------------------------------------------------------
# Breakdown sums to same total as estimate()
# ------------------------------------------------------------------
section("FLOPsEstimator — breakdown() sums to estimate()")

total_from_estimate   = estimator.estimate(arch_spec)
total_from_breakdown  = sum(f for _, f in estimator.breakdown(arch_spec))
print(f"  estimate()   : {total_from_estimate:,}")
print(f"  breakdown()  : {total_from_breakdown:,}")
assert total_from_estimate == total_from_breakdown
print("  PASS")


# ------------------------------------------------------------------
# RewardCombiner — basic reward properties
# ------------------------------------------------------------------
section("RewardCombiner — reward at target FLOPs equals accuracy")

rc = RewardCombiner(flops_target=50e6, alpha=0.07, norm_warmup=1)

# When flops == flops_target, cost_ratio = 1.0, so R = acc * 1.0^alpha = acc
norm_r, raw_r = rc.compute(accuracy=0.80, flops=50e6)
print(f"  accuracy=0.80, flops=target  ->  raw_reward={raw_r:.6f}")
assert abs(raw_r - 0.80) < 1e-6, \
    f"Expected raw_reward == accuracy when flops == target, got {raw_r}"
print("  PASS")


# ------------------------------------------------------------------
# Cheaper arch gets higher reward than expensive arch (same accuracy)
# ------------------------------------------------------------------
section("RewardCombiner — cheaper arch rewarded more")

rc2 = RewardCombiner(flops_target=50e6, alpha=0.07, norm_warmup=1)
_, cheap_r = rc2.compute(accuracy=0.80, flops=25e6)   # half the budget
_, expen_r = rc2.compute(accuracy=0.80, flops=100e6)  # double the budget
print(f"  25M FLOPs raw reward  : {cheap_r:.6f}")
print(f"  100M FLOPs raw reward : {expen_r:.6f}")
assert cheap_r > expen_r, "Cheaper arch should receive higher reward"
print("  PASS")


# ------------------------------------------------------------------
# Higher alpha means stronger cost penalty
# ------------------------------------------------------------------
section("RewardCombiner — higher alpha = stronger cost penalty")

rc_mild   = RewardCombiner(flops_target=50e6, alpha=0.07,  norm_warmup=1)
rc_strong = RewardCombiner(flops_target=50e6, alpha=0.30,  norm_warmup=1)

_, mild_r   = rc_mild.compute(accuracy=0.80,   flops=200e6)
_, strong_r = rc_strong.compute(accuracy=0.80, flops=200e6)
print(f"  alpha=0.07  raw reward : {mild_r:.6f}")
print(f"  alpha=0.30  raw reward : {strong_r:.6f}")
assert strong_r < mild_r, \
    "Stronger alpha should penalise expensive archs more"
print("  PASS")


# ------------------------------------------------------------------
# Normalisation — after warmup, mean ~ 0 and std ~ 1
# ------------------------------------------------------------------
section("RewardCombiner — normalised rewards have mean~0, std~1")

import random
rc3 = RewardCombiner(flops_target=50e6, alpha=0.07, norm_warmup=5)

norm_rewards = []
for _ in range(50):
    acc   = random.uniform(0.5, 0.9)
    flops = random.uniform(20e6, 100e6)
    norm_r, _ = rc3.compute(acc, flops)
    norm_rewards.append(norm_r)

# Only look at rewards after warmup
post_warmup = norm_rewards[5:]
mean = sum(post_warmup) / len(post_warmup)
std  = (sum((r - mean)**2 for r in post_warmup) / len(post_warmup)) ** 0.5

print(f"  Post-warmup mean : {mean:.4f}  (expected ~0)")
print(f"  Post-warmup std  : {std:.4f}  (expected ~1)")
assert abs(mean) < 0.3, f"Mean too far from 0: {mean}"
assert 0.5 < std < 1.5, f"Std out of expected range: {std}"
print("  PASS")


# ------------------------------------------------------------------
# Stats tracking
# ------------------------------------------------------------------
section("RewardCombiner — stats() and properties")

stats = rc3.stats()
print(f"  n_episodes      : {stats['n_episodes']}")
print(f"  mean_raw_reward : {stats['mean_raw_reward']:.4f}")
print(f"  best_raw_reward : {stats['best_raw_reward']:.4f}")
assert stats['n_episodes'] == 50
assert stats['best_raw_reward'] >= stats['mean_raw_reward']
print("  PASS")


# ------------------------------------------------------------------
# Reset clears history
# ------------------------------------------------------------------
section("RewardCombiner — reset() clears history")

rc3.reset()
assert rc3.n_episodes == 0
assert rc3.mean_reward == 0.0
print("  PASS")


print()
print("All tests PASSED")