# Search loop definition
"""
The outer RL loop that ties every module together.

One episode = one full round-trip:
  1. Controller samples an architecture (arch_spec, log_probs, entropies)
  2. Evaluator scores it on the proxy val set  -> accuracy
  3. FLOPs estimator measures its cost         -> flops
  4. RewardCombiner turns those into a scalar  -> reward
  5. Baseline is updated
  6. REINFORCE loss + entropy bonus computed
  7. Controller weights updated via Adam
  8. Architecture logged in the pool

The supernet weights are NOT updated during search — they stay frozen
from pretraining.  Only the controller learns.
"""

import time
import torch
import torch.nn as nn

from nas_rl.controller.baseline import ExponentialBaseline
from nas_rl.search.architecture_pool import ArchitecturePool


class SearchLoop:
    """
    Parameters
    ----------
    supernet        : Supernet         frozen after pretraining
    controller      : LSTMController   the RL agent being trained
    evaluator       : Evaluator        scores archs on the val set
    reward_combiner : RewardCombiner   accuracy + cost -> scalar reward
    cost_estimator  : FLOPsEstimator   FLOPs for a given arch_spec
    device          : torch.device

    ctrl_lr         : float   Adam LR for the controller (default 3e-3)
    entropy_coeff   : float   weight of the entropy bonus (default 0.01)
                              Higher values = more exploration.
    baseline_decay  : float   EMA decay for the reward baseline (0.95)
    pool_capacity   : int     max unique archs to remember (200)
    """

    def __init__(
        self,
        supernet,
        controller,
        evaluator,
        reward_combiner,
        cost_estimator,
        device,
        ctrl_lr       = 3e-3,
        entropy_coeff = 0.01,
        baseline_decay= 0.95,
        pool_capacity = 200,
    ):
        self.supernet        = supernet
        self.controller      = controller
        self.evaluator       = evaluator
        self.reward_combiner = reward_combiner
        self.cost_estimator  = cost_estimator
        self.device          = device
        self.entropy_coeff   = entropy_coeff

        # Freeze supernet — only controller trains during search
        for p in self.supernet.parameters():
            p.requires_grad_(False)

        self.ctrl_optimizer = torch.optim.Adam(
            controller.parameters(),
            lr=ctrl_lr,
            weight_decay=1e-4,
        )

        self.baseline  = ExponentialBaseline(decay=baseline_decay)
        self.arch_pool = ArchitecturePool(capacity=pool_capacity)

        self._episode = 0   # global episode counter

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------

    def run_episode(self):
        """
        Execute one full NAS-RL episode.

        Returns a stats dict with all metrics for logging.
        """
        t0 = time.time()

        # 1. Controller samples one architecture
        arch_spec, log_probs, entropies = self.controller(self.device)

        # 2. Score it on the proxy val set (no gradients into supernet)
        accuracy = self.evaluator.evaluate(arch_spec)

        # 3. Estimate FLOPs
        flops = self.cost_estimator.estimate(arch_spec)

        # 4. Compute reward (normalised scalar)
        reward, raw_reward = self.reward_combiner.compute(accuracy, flops)

        # 5. Update baseline
        baseline_val = self.baseline.update(raw_reward)

        # 6. REINFORCE loss
        #    advantage = reward - baseline  (reduces gradient variance)
        #    policy loss = -sum( log_prob(a_t) * advantage )
        advantage    = reward - baseline_val
        policy_loss  = -(log_probs * advantage).sum()

        # Entropy bonus: encourages exploration by penalising low entropy.
        # Subtracted because we minimise loss (higher entropy = lower loss).
        entropy_loss = -self.entropy_coeff * entropies.sum()

        total_loss = policy_loss + entropy_loss

        # 7. Update controller weights
        self.ctrl_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.controller.parameters(), 5.0)
        self.ctrl_optimizer.step()

        # 8. Store in pool
        self._episode += 1
        is_new = self.arch_pool.add(
            arch_spec, accuracy, flops, reward,
            episode=self._episode,
        )

        elapsed = time.time() - t0

        return {
            'episode':      self._episode,
            'accuracy':     accuracy,
            'flops':        flops,
            'raw_reward':   raw_reward,
            'reward':       reward,
            'advantage':    advantage,
            'policy_loss':  policy_loss.item(),
            'entropy':      entropies.mean().item(),
            'total_loss':   total_loss.item(),
            'baseline':     baseline_val,
            'is_new_arch':  is_new,
            'elapsed_s':    elapsed,
        }

    # ------------------------------------------------------------------
    # Full search run
    # ------------------------------------------------------------------

    def run(self, n_episodes=200, log_every=10):
        """
        Run the search loop for n_episodes episodes.

        Prints a summary every log_every episodes.
        Returns the top-5 architectures found.
        """
        print(f"\nStarting NAS search — {n_episodes} episodes")
        print(f"{'Ep':>5}  {'Acc':>7}  {'FLOPs':>8}  "
              f"{'RawR':>7}  {'NormR':>7}  {'Entr':>6}  "
              f"{'New':>4}  {'Time':>6}")
        print("-" * 65)

        for ep in range(n_episodes):
            stats = self.run_episode()

            if (ep + 1) % log_every == 0 or ep == 0:
                print(
                    f"{stats['episode']:5d}  "
                    f"{stats['accuracy']*100:6.2f}%  "
                    f"{stats['flops']/1e6:7.1f}M  "
                    f"{stats['raw_reward']:7.4f}  "
                    f"{stats['reward']:7.4f}  "
                    f"{stats['entropy']:6.3f}  "
                    f"{'yes' if stats['is_new_arch'] else 'no':>4}  "
                    f"{stats['elapsed_s']:5.1f}s"
                )

        print("-" * 65)
        print(f"Search complete.  Unique archs found: {len(self.arch_pool)}")
        self.arch_pool.summary()

        return self.arch_pool.top_k(5)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path):
        """Save controller weights and search state."""
        torch.save({
            'episode':          self._episode,
            'controller':       self.controller.state_dict(),
            'ctrl_optimizer':   self.ctrl_optimizer.state_dict(),
            'baseline_value':   self.baseline.value,
            'reward_history':   self.reward_combiner._history,
            'arch_pool':        self.arch_pool.pool,
        }, path)
        print(f"Checkpoint saved -> {path}")

    def load_checkpoint(self, path):
        """Resume from a saved checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self._episode = ckpt['episode']
        self.controller.load_state_dict(ckpt['controller'])
        self.ctrl_optimizer.load_state_dict(ckpt['ctrl_optimizer'])
        self.baseline.value = ckpt['baseline_value']
        self.reward_combiner._history = ckpt['reward_history']
        # Rebuild pool from saved entries
        for entry in ckpt['arch_pool']:
            self.arch_pool.add(
                entry['arch'], entry['accuracy'],
                entry['flops'], entry['reward'], entry['episode'],
            )
        print(f"Resumed from episode {self._episode}")