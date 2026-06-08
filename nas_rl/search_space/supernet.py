# Supernet definition
import random
import torch
import torch.nn as nn
from nas_rl.search_space.ops import IDX_TO_OP
from nas_rl.search_space.cell import Cell, ReLUConvBN


class Supernet(nn.Module):
    """
    Full weight-sharing supernet built from stacked cells.

    Layout (default: 8 cells on CIFAR-10 32x32):
      stem -> cell_0 .. cell_7 -> global avg pool -> linear classifier

    Reduction cells at indices n_cells//3 and 2*n_cells//3 halve spatial
    dims and double channel count.  All other cells are normal cells.

    forward(x, arch_spec) routes every cell via arch_spec — a list of dicts,
    one per cell, each mapping edge keys to op name strings.  The same
    supernet handles any architecture in the search space.
    """

    def __init__(self, C_init=16, n_cells=8, n_nodes=4, n_classes=10):
        super().__init__()
        self.n_cells  = n_cells
        self.n_nodes  = n_nodes
        self.n_classes = n_classes

        C_stem = C_init * 3
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_stem, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_stem),
        )

        self.reduction_indices = {n_cells // 3, 2 * n_cells // 3}

        cells       = []
        C_prev_prev = C_stem
        C_prev      = C_stem
        C_curr      = C_init
        prev_reduction = False   # tracks whether the previous cell was a reduction

        for i in range(n_cells):
            reduction = (i in self.reduction_indices)
            if reduction:
                C_curr *= 2

            cell = Cell(
                n_nodes      = n_nodes,
                C_prev_prev  = C_prev_prev,
                C_prev       = C_prev,
                C            = C_curr,
                reduction    = reduction,
                prev_reduction = prev_reduction,
            )
            cells.append(cell)

            C_prev_prev    = C_prev
            C_prev         = n_nodes * C_curr   # concat of n_nodes outputs
            prev_reduction = reduction

        self.cells       = nn.ModuleList(cells)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Linear(C_prev, n_classes)

    def forward(self, x, arch_spec):
        """
        x        : (B, 3, H, W)
        arch_spec: list[dict]  length == n_cells
        returns  : (B, n_classes) logits
        """
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, arch_spec[i])
        out = self.global_pool(s1).flatten(1)
        return self.classifier(out)

    def random_arch(self):
        """
        Uniformly random architecture spec — used during supernet pretraining
        so all ops are trained without systematic bias.
        """
        arch_spec = []
        for _ in range(self.n_cells):
            cell_ops = {
                key: random.choice(IDX_TO_OP)
                for key in Cell.edge_keys(self.n_nodes)
            }
            arch_spec.append(cell_ops)
        return arch_spec

    def cell_resolutions(self, input_size=32):
        """
        Spatial resolution at the input of each cell.
        Used by the FLOPs estimator.
        """
        res = input_size
        resolutions = []
        for i in range(self.n_cells):
            resolutions.append(res)
            if i in self.reduction_indices:
                res //= 2
        return resolutions