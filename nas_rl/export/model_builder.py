# Model builder definition
"""
Builds a clean standalone model from a winning architecture spec.

During search, architectures run through the supernet — a large shared
network with ModuleDict routing on every edge.  That's efficient for
search but wasteful for deployment: you're loading weights for all ops
even though only one per edge is ever used.

FixedCell and StandaloneModel strip all of that out.  The result is a
plain nn.Module that only contains the ops actually chosen, with no
routing logic, no ModuleDict overhead, and no supernet dependency.
It's also faster to run and easier to export to CoreML / ONNX.
"""

import torch
import torch.nn as nn
from nas_rl.search_space.ops import OPS, FactorizedReduce
from nas_rl.search_space.cell import ReLUConvBN


class FixedCell(nn.Module):
    """
    A cell with one fixed op per edge — no routing, no ModuleDict.

    Built from a cell_spec dict (edge_key -> op_name) produced by the
    controller.  Only instantiates the chosen ops, so it's much smaller
    than the supernet cell.

    forward(s0, s1) needs no arch_weights — the op choice is baked in
    at construction time.
    """

    def __init__(self, cell_spec, n_nodes, C_prev_prev, C_prev, C,
                 reduction, prev_reduction):
        super().__init__()
        self.n_nodes   = n_nodes
        self.reduction = reduction
        stride = 2 if reduction else 1

        if prev_reduction:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        # Store ops in a ModuleDict keyed by edge key — simple and
        # compatible with torch.jit.trace / ONNX export.
        self.ops = nn.ModuleDict()
        for i in range(n_nodes):
            for j in range(i + 2):
                key         = f'{j}_{i+2}'
                op_name     = cell_spec[key]
                edge_stride = stride if j < 2 else 1
                self.ops[key] = OPS[op_name](C, edge_stride)

        self.n_nodes = n_nodes

    def forward(self, s0, s1):
        """No arch_weights argument — op choice is baked in at build time."""
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        # Mirror the exact loop from cell.py so indexing is identical
        for i in range(self.n_nodes):
            node_inputs = []
            for j in range(i + 2):
                key = f'{j}_{i+2}'
                node_inputs.append(self.ops[key](states[j]))
            states.append(sum(node_inputs))

        return torch.cat(states[2:], dim=1)


class StandaloneModel(nn.Module):
    """
    A complete deployable model built from a winning arch_spec.

    Identical macro structure to the supernet (same stem, same cell
    stacking, same classifier) but uses FixedCell so only the chosen
    ops exist in memory.

    Parameters
    ----------
    arch_spec   : list[dict]  — from controller or arch_pool.top_k()
    C_init      : int         — must match the supernet used during search
    n_cells     : int
    n_nodes     : int
    n_classes   : int
    """

    def __init__(self, arch_spec, C_init=16, n_cells=8,
                 n_nodes=4, n_classes=10):
        super().__init__()
        self.n_cells  = n_cells
        self.n_nodes  = n_nodes

        C_stem = C_init * 3
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_stem, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_stem),
        )

        reduction_indices = {n_cells // 3, 2 * n_cells // 3}

        cells          = []
        C_prev_prev    = C_stem
        C_prev         = C_stem
        C_curr         = C_init
        prev_reduction = False

        for i in range(n_cells):
            reduction = (i in reduction_indices)
            if reduction:
                C_curr *= 2

            cell = FixedCell(
                cell_spec      = arch_spec[i],
                n_nodes        = n_nodes,
                C_prev_prev    = C_prev_prev,
                C_prev         = C_prev,
                C              = C_curr,
                reduction      = reduction,
                prev_reduction = prev_reduction,
            )
            cells.append(cell)

            C_prev_prev    = C_prev
            C_prev         = n_nodes * C_curr
            prev_reduction = reduction

        self.cells       = nn.ModuleList(cells)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Linear(C_prev, n_classes)

    def forward(self, x):
        """Clean forward pass — no arch_spec argument needed."""
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pool(s1).flatten(1)
        return self.classifier(out)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def transfer_weights(supernet, standalone, arch_spec):
    """
    Copy matching weights from the supernet into the standalone model.

    The supernet has already trained op weights — transferring them
    gives the standalone model a warm start so final fine-tuning
    converges faster than training from scratch.

    Strategy:
      1. Try direct key match (stem, classifier, preprocessing layers)
      2. For cell ops: remap standalone key -> supernet key by inserting
         the op name from arch_spec into the key path

    Supernet op key format:
        cells.{i}.edges.{edge_key}.{op_name}.{rest}
    Standalone op key format:
        cells.{i}.ops.{edge_key}.{rest}
    """
    snet_state  = supernet.state_dict()
    stand_state = standalone.state_dict()

    copied  = 0
    skipped = 0

    for key in stand_state:
        # 1. Direct match (stem, classifier, preprocess layers)
        if key in snet_state and snet_state[key].shape == stand_state[key].shape:
            stand_state[key] = snet_state[key].clone()
            copied += 1
            continue

        # 2. Remap cell op keys
        # standalone: cells.{ci}.ops.{edge_key}.{rest...}
        parts = key.split('.')
        if (len(parts) >= 5
                and parts[0] == 'cells'
                and parts[2] == 'ops'):
            cell_idx  = int(parts[1])
            edge_key  = parts[3]
            rest      = '.'.join(parts[4:])
            op_name   = arch_spec[cell_idx].get(edge_key)
            if op_name is not None:
                snet_key = f'cells.{cell_idx}.edges.{edge_key}.{op_name}.{rest}'
                if (snet_key in snet_state
                        and snet_state[snet_key].shape == stand_state[key].shape):
                    stand_state[key] = snet_state[snet_key].clone()
                    copied += 1
                    continue

        skipped += 1

    standalone.load_state_dict(stand_state)
    print(f"  Weight transfer: {copied} tensors copied, {skipped} skipped")
    return standalone