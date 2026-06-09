# Cost estimator definition
"""
FLOPs estimator for NAS architectures.

Computes FLOPs analytically from a lookup table built at init time —
much faster than running a real profiler every episode.

FLOPs formula used throughout:
  Conv (k x k, C_in -> C_out, H x W output):
      2 * C_in * C_out * k * k * H * W
  The factor of 2 accounts for multiply-accumulate (1 mult + 1 add).

Depthwise conv (groups=C):
      2 * C * k * k * H * W   (C_in == C_out == groups == C)

Pointwise conv (1x1, C -> C):
      2 * C * C * H * W

Pooling (k x k, C channels):
      C * k * k * H * W       (comparisons only, no multiplies)
"""

from nas_rl.search_space.ops import IDX_TO_OP


def _conv_flops(C_in, C_out, k, H, W):
    return 2 * C_in * C_out * k * k * H * W


def _pool_flops(C, k, H, W):
    return C * k * k * H * W


def _op_flops(op_name, C, H, W):
    """Analytical FLOPs for one op applied to a (C, H, W) feature map."""
    if op_name == 'conv3x3':
        return _conv_flops(C, C, 3, H, W)
    elif op_name == 'conv1x1':
        return _conv_flops(C, C, 1, H, W)
    elif op_name == 'sep3x3':
        # depthwise 3x3 + pointwise 1x1
        return _conv_flops(1, 1, 3, H, W) * C + _conv_flops(C, C, 1, H, W)
    elif op_name == 'sep5x5':
        # depthwise 5x5 + pointwise 1x1
        return _conv_flops(1, 1, 5, H, W) * C + _conv_flops(C, C, 1, H, W)
    elif op_name == 'maxpool':
        return _pool_flops(C, 3, H, W)
    elif op_name == 'avgpool':
        return _pool_flops(C, 3, H, W)
    elif op_name in ('skip', 'zero'):
        return 0   # no compute
    else:
        raise ValueError(f"Unknown op: {op_name}")


class FLOPsEstimator:
    """
    Estimates total FLOPs for a given arch_spec.

    The lookup table is keyed by (spatial_resolution, op_name) because
    FLOPs depend on the feature map size, which changes after each
    reduction cell.

    Channel counts per cell are also tracked — they grow after each
    reduction cell as the supernet doubles channels to compensate for
    halved spatial dims.

    Usage
    -----
    estimator = FLOPsEstimator(supernet)
    flops = estimator.estimate(arch_spec)
    """

    def __init__(self, supernet, input_size=32):
        self.table = self._build_table(supernet, input_size)

    def _build_table(self, supernet, input_size):
        """
        Walk through the supernet's cell structure to record the spatial
        resolution and channel count at each cell, then compute FLOPs
        for every op at every (cell, resolution, channel) combination.

        Returns a dict: cell_idx -> {op_name -> flops_per_edge}
        Each edge in the cell will have n_nodes uses on average, but we
        store per-op FLOPs and multiply by the actual edge count at
        estimate time.
        """
        table = {}
        resolutions = supernet.cell_resolutions(input_size)

        # Reconstruct channel counts per cell (mirrors supernet.__init__)
        from nas_rl.search_space.supernet import Supernet
        C_init = supernet.classifier.in_features  # C_prev after last cell
        # Walk forward to get per-cell C_curr
        # Re-derive from reduction_indices and n_cells
        n_cells = supernet.n_cells
        n_nodes = supernet.n_nodes
        reduction_indices = supernet.reduction_indices

        # Find C_init by reversing: C_prev_last = n_nodes * C_curr_last
        # and C_curr doubles at each reduction.  Count reductions after start.
        # Simpler: just replay the supernet channel progression.
        C_curr = None
        # The stem outputs C_init_raw * 3; then cells start at C_init_raw
        # We can extract C_init_raw from supernet.stem[0].out_channels / 3
        C_init_raw = supernet.stem[0].out_channels // 3
        C_curr = C_init_raw
        cell_channels = []
        for i in range(n_cells):
            if i in reduction_indices:
                C_curr *= 2
            cell_channels.append(C_curr)

        for cell_idx in range(n_cells):
            res = resolutions[cell_idx]
            C   = cell_channels[cell_idx]
            # After a reduction cell, output spatial is res//2
            out_res = res // 2 if cell_idx in reduction_indices else res
            table[cell_idx] = {
                op_name: _op_flops(op_name, C, out_res, out_res)
                for op_name in IDX_TO_OP
            }

        return table

    def estimate(self, arch_spec):
        """
        Sum FLOPs across every edge in every cell for the given arch_spec.

        Parameters
        ----------
        arch_spec : list[dict]  — one dict per cell, edge_key -> op_name

        Returns
        -------
        total_flops : int
        """
        total = 0
        for cell_idx, cell_ops in enumerate(arch_spec):
            cell_table = self.table[cell_idx]
            for op_name in cell_ops.values():
                total += cell_table[op_name]
        return total

    def breakdown(self, arch_spec):
        """
        Per-cell FLOPs breakdown — useful for debugging and visualisation.
        Returns a list of (cell_idx, total_flops_for_cell) tuples.
        """
        result = []
        for cell_idx, cell_ops in enumerate(arch_spec):
            cell_table = self.table[cell_idx]
            cell_flops = sum(cell_table[op] for op in cell_ops.values())
            result.append((cell_idx, cell_flops))
        return result