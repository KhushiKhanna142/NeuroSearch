# Architecture pool definition
"""
Stores every architecture evaluated during search with its metrics.

Provides:
  - Deduplication: the controller sometimes samples the same architecture
    twice, especially late in search when it has converged.  Re-evaluating
    duplicates wastes episodes without adding information.
  - Top-K retrieval: after search, rank candidates by reward (or accuracy)
    and return the best ones for final fine-tuning.
  - History: full log of every unique architecture for analysis.
"""


class ArchitecturePool:
    """
    Parameters
    ----------
    capacity : int
        Maximum number of unique architectures to store.
        Once full, new entries replace the worst-reward entry if the
        new reward is better (keeps the pool as the top-N seen so far).
        Set to None for unlimited storage.
    """

    def __init__(self, capacity=200):
        self.capacity = capacity
        self.pool     = []          # list of entry dicts
        self._hashes  = set()       # for O(1) duplicate detection

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash(self, arch_spec):
        """
        Deterministic string key for an arch_spec.
        Sorts each cell's items so key is order-independent within a cell.
        """
        return str([
            sorted(cell.items())
            for cell in arch_spec
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, arch_spec, accuracy, flops, reward, episode=None):
        """
        Add a new architecture to the pool.

        Returns True if added, False if it was a duplicate.

        Parameters
        ----------
        arch_spec : list[dict]
        accuracy  : float   top-1 accuracy on proxy val set
        flops     : int     estimated FLOPs
        reward    : float   normalised reward from RewardCombiner
        episode   : int     episode number (for logging)
        """
        h = self._hash(arch_spec)
        if h in self._hashes:
            return False   # duplicate — skip

        entry = {
            'arch':     arch_spec,
            'accuracy': accuracy,
            'flops':    flops,
            'reward':   reward,
            'episode':  episode,
            'hash':     h,
        }

        if self.capacity is None or len(self.pool) < self.capacity:
            self.pool.append(entry)
            self._hashes.add(h)
        else:
            # Replace the worst entry if this one is better
            worst_idx = min(range(len(self.pool)),
                            key=lambda i: self.pool[i]['reward'])
            if reward > self.pool[worst_idx]['reward']:
                self._hashes.discard(self.pool[worst_idx]['hash'])
                self.pool[worst_idx] = entry
                self._hashes.add(h)
            else:
                return False   # worse than everything in pool, skip

        return True

    def top_k(self, k, sort_by='reward'):
        """
        Return the top-k entries sorted by the given metric (descending).

        sort_by options: 'reward', 'accuracy', 'flops' (ascending for flops)
        """
        assert sort_by in ('reward', 'accuracy', 'flops')
        reverse = (sort_by != 'flops')   # flops: smaller is better
        return sorted(self.pool, key=lambda e: e[sort_by], reverse=reverse)[:k]

    def best(self):
        """Single best entry by reward."""
        if not self.pool:
            return None
        return max(self.pool, key=lambda e: e['reward'])

    def __len__(self):
        return len(self.pool)

    def summary(self):
        """Print a compact table of the top-10 entries."""
        top = self.top_k(10)
        print(f"\n{'Rank':>4}  {'Episode':>7}  {'Accuracy':>9}  "
              f"{'FLOPs(M)':>9}  {'Reward':>8}")
        print("-" * 46)
        for rank, e in enumerate(top, 1):
            ep  = e['episode'] if e['episode'] is not None else '—'
            print(f"{rank:4d}  {str(ep):>7}  {e['accuracy']*100:8.2f}%  "
                  f"{e['flops']/1e6:9.2f}  {e['reward']:8.4f}")