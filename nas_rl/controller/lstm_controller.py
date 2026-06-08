# LSTM Controller definition
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from nas_rl.search_space.ops import IDX_TO_OP, N_OPS
from nas_rl.search_space.cell import Cell


class LSTMController(nn.Module):
    """
    RL controller that sequentially samples an architecture.

    For every edge in every cell the LSTM makes one decision:
    which op to place on that edge.  The hidden state carries
    context so later choices are informed by earlier ones.

    One forward() call = one complete architecture sample.

    Returns
    -------
    arch_spec  : list[dict]  — one dict per cell, edge_key -> op_name
    log_probs  : Tensor (n_decisions,)  — log P of each sampled token
    entropies  : Tensor (n_decisions,)  — entropy of each decision dist
    """

    def __init__(self, n_cells=8, n_nodes=4, hidden_dim=64):
        super().__init__()
        self.n_cells     = n_cells
        self.n_nodes     = n_nodes
        self.hidden_dim  = hidden_dim
        self.edge_keys   = Cell.edge_keys(n_nodes)   # 14 keys, fixed order
        self.n_edges     = len(self.edge_keys)        # 14
        self.n_decisions = n_cells * self.n_edges     # 112

        # op index (scalar) -> dense vector of shape (hidden_dim,)
        self.embedding = nn.Embedding(N_OPS, hidden_dim)

        # One LSTM step per decision
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)

        # hidden state -> logits over op vocabulary
        self.op_head = nn.Linear(hidden_dim, N_OPS)

        # Learned start token: shape (1, hidden_dim) — feeds the first step
        self.start_token = nn.Parameter(torch.zeros(1, hidden_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.op_head.weight,   -0.1, 0.1)
        nn.init.zeros_(self.op_head.bias)

    def forward(self, device):
        """
        Stochastically sample one complete architecture.

        Returns
        -------
        arch_spec : list[dict]
        log_probs : Tensor (n_decisions,)
        entropies : Tensor (n_decisions,)
        """
        h = torch.zeros(1, self.hidden_dim, device=device)
        c = torch.zeros(1, self.hidden_dim, device=device)

        # x is always shape (1, hidden_dim) throughout the loop
        x = self.start_token.to(device)

        arch_spec = []
        log_probs = []
        entropies = []

        for _ in range(self.n_cells):
            cell_ops = {}

            for key in self.edge_keys:
                # --- one LSTM step ---
                h, c = self.lstm(x, (h, c))            # h: (1, hidden_dim)

                logits = self.op_head(h)                # (1, N_OPS)
                probs  = F.softmax(logits, dim=-1)      # (1, N_OPS)
                dist   = Categorical(probs)             # batch size 1

                op_idx = dist.sample()                  # shape: (1,)

                log_probs.append(dist.log_prob(op_idx).squeeze())  # scalar
                entropies.append(dist.entropy().squeeze())         # scalar

                # op_idx is shape (1,) so embedding returns (1, hidden_dim)
                # — no unsqueeze needed; that's exactly the shape lstm expects
                cell_ops[key] = IDX_TO_OP[op_idx.item()]
                x = self.embedding(op_idx)              # (1, hidden_dim)

            arch_spec.append(cell_ops)

        return (
            arch_spec,
            torch.stack(log_probs),     # (n_decisions,)
            torch.stack(entropies),     # (n_decisions,)
        )

    def greedy_arch(self, device):
        """
        Deterministic: always pick the argmax op.
        Useful for inspecting controller convergence.
        """
        h = torch.zeros(1, self.hidden_dim, device=device)
        c = torch.zeros(1, self.hidden_dim, device=device)
        x = self.start_token.to(device)

        arch_spec = []
        with torch.no_grad():
            for _ in range(self.n_cells):
                cell_ops = {}
                for key in self.edge_keys:
                    h, c   = self.lstm(x, (h, c))
                    logits = self.op_head(h)
                    op_idx = logits.argmax(dim=-1)      # shape: (1,)
                    cell_ops[key] = IDX_TO_OP[op_idx.item()]
                    x = self.embedding(op_idx)          # (1, hidden_dim)
                arch_spec.append(cell_ops)
        return arch_spec