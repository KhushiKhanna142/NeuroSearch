# Cell definition
import torch
import torch.nn as nn
from nas_rl.search_space.ops import OPS, IDX_TO_OP, FactorizedReduce


class ReLUConvBN(nn.Module):
    """ReLU -> Conv2d -> BatchNorm2d  (preprocessing block inside each cell)"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        return self.op(x)


class Cell(nn.Module):
    """
    A single cell in the search space — a small DAG with n_nodes
    intermediate nodes plus two inputs from the previous two cells.

    Edge key convention: edge from node j to node i is keyed f'{j}_{i+2}'
    because cell inputs occupy slots 0 and 1, intermediates start at 2.

    preprocess0/preprocess1 align the two incoming cell outputs to C channels.
    If the previous cell was a reduction (prev_reduction=True), then s0 (from
    two cells back) still has 2x the spatial size of s1 — so preprocess0 must
    also halve spatial dims via FactorizedReduce instead of a plain 1x1 conv.

    forward(s0, s1, arch_weights) routes any architecture through the cell
    without retraining — this is the weight-sharing mechanism.
    """

    def __init__(self, n_nodes, C_prev_prev, C_prev, C,
                 reduction, prev_reduction):
        super().__init__()
        self.n_nodes = n_nodes
        self.reduction = reduction
        stride = 2 if reduction else 1

        # preprocess0: if the *previous* cell was a reduction, s0 still has
        # the larger spatial size and needs to be downsampled to match s1.
        if prev_reduction:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        # One ModuleDict per edge; each contains every candidate op
        self.edges = nn.ModuleDict()
        for i in range(n_nodes):
            for j in range(i + 2):
                key = f'{j}_{i + 2}'
                # Edges from the two cell inputs (j<2) get the cell stride;
                # edges between intermediate nodes always use stride=1
                edge_stride = stride if j < 2 else 1
                self.edges[key] = nn.ModuleDict({
                    op_name: OPS[op_name](C, edge_stride)
                    for op_name in IDX_TO_OP
                })

    def forward(self, s0, s1, arch_weights):
        """
        s0, s1      : outputs of the two previous cells
        arch_weights: dict  edge_key -> op_name_string
        returns     : (B, C * n_nodes, H', W')
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        for i in range(self.n_nodes):
            node_inputs = []
            for j in range(i + 2):
                key = f'{j}_{i + 2}'
                op_name = arch_weights[key]
                node_inputs.append(self.edges[key][op_name](states[j]))
            states.append(sum(node_inputs))

        return torch.cat(states[2:], dim=1)

    @staticmethod
    def edge_keys(n_nodes):
        """All edge keys for a cell with n_nodes intermediate nodes."""
        keys = []
        for i in range(n_nodes):
            for j in range(i + 2):
                keys.append(f'{j}_{i + 2}')
        return keys