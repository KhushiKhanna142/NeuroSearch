# Reward baseline tracking
class ExponentialBaseline:
    """
    Exponential moving average of recent rewards.

    Subtracting this from each reward before computing the REINFORCE
    gradient substantially reduces variance without introducing bias —
    the baseline doesn't depend on the current action.

    decay=0.95 means roughly the last 20 rewards are weighted heavily.
    A single outlier reward won't destabilize the gradient signal.
    """

    def __init__(self, decay=0.95):
        self.decay = decay
        self.value = None          # None until the first reward arrives

    def update(self, reward):
        """Update EMA with a new reward and return the updated value."""
        if self.value is None:
            self.value = reward    # cold start: first reward sets the baseline
        else:
            self.value = self.decay * self.value + (1 - self.decay) * reward
        return self.value

    def get(self):
        """Current baseline value (0.0 before any rewards have arrived)."""
        return self.value if self.value is not None else 0.0

    def reset(self):
        """Clear state — useful when starting a new search run."""
        self.value = None