# Reward combiner definition
"""
Reward combiner for NAS with RL.

Combines accuracy and FLOPs into a single scalar reward using the
hardware-aware formula from MNasNet / ProxylessNAS:

    R = accuracy * (flops_target / flops) ^ alpha

alpha controls how hard the cost constraint is enforced:
  - alpha = 0.0  : pure accuracy, cost ignored
  - alpha = 0.07 : mild penalty (default — good starting point)
  - alpha = 0.2+ : strong penalty, controller heavily prefers cheap archs

The raw reward is then z-score normalised across all rewards seen so far
in the current search run.  This is critical for REINFORCE stability:
  - Early episodes have high variance; normalising prevents overreaction
  - Keeps the gradient magnitude consistent regardless of reward scale
  - Makes the entropy coefficient (exploration bonus) easier to tune
"""


class RewardCombiner:
    """
    Computes and normalises rewards for the RL controller.

    Parameters
    ----------
    flops_target : float
        The FLOPs budget you want to optimise toward.
        Architectures at exactly this cost get a cost_ratio of 1.0
        (no penalty or bonus).  Architectures below it get a bonus;
        above it get penalised.
        Default 50M FLOPs — reasonable for a small CIFAR-10 model.

    alpha : float
        Cost penalty exponent.  0.07 is a mild nudge toward efficiency.
        Increase to 0.15-0.2 if the controller consistently ignores the
        budget.

    norm_warmup : int
        Number of episodes to collect before normalising.  During warmup
        the raw reward is returned as-is so the baseline has time to
        stabilise before normalisation kicks in.
    """

    def __init__(self, flops_target=50e6, alpha=0.07, norm_warmup=10):
        self.flops_target = flops_target
        self.alpha        = alpha
        self.norm_warmup  = norm_warmup

        self._history = []   # raw rewards across all episodes

    def compute(self, accuracy, flops):
        """
        Compute the normalised reward for one episode.

        Parameters
        ----------
        accuracy : float  — top-1 accuracy in [0, 1]
        flops    : int    — FLOPs for the sampled architecture

        Returns
        -------
        normalised_reward : float
        raw_reward        : float  (for logging)
        """
        # Cost ratio: > 1 if cheaper than target, < 1 if more expensive
        cost_ratio = self.flops_target / max(flops, 1)
        raw_reward = accuracy * (cost_ratio ** self.alpha)

        self._history.append(raw_reward)

        # Warmup: return raw reward until we have enough history
        if len(self._history) < self.norm_warmup:
            return raw_reward, raw_reward

        normalised = self._normalise(raw_reward)
        return normalised, raw_reward

    def _normalise(self, reward):
        """Z-score normalise using running mean and std of all rewards so far."""
        n    = len(self._history)
        mean = sum(self._history) / n
        var  = sum((r - mean) ** 2 for r in self._history) / n
        std  = var ** 0.5
        return (reward - mean) / (std + 1e-8)

    def reset(self):
        """Clear reward history — call this when starting a new search run."""
        self._history = []

    @property
    def n_episodes(self):
        return len(self._history)

    @property
    def mean_reward(self):
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    @property
    def best_raw_reward(self):
        return max(self._history) if self._history else 0.0

    def stats(self):
        """Summary dict for logging."""
        return {
            'n_episodes':       self.n_episodes,
            'mean_raw_reward':  self.mean_reward,
            'best_raw_reward':  self.best_raw_reward,
            'flops_target':     self.flops_target,
            'alpha':            self.alpha,
        }