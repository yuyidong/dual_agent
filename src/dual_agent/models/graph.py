from __future__ import annotations

import torch
from torch import nn


def normalize_adjacency(adjacency: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """Symmetric normalization for a dense adjacency matrix."""
    if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
        raise ValueError("adjacency must have shape [nodes, nodes]")
    adj = adjacency.float()
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    degree = adj.sum(dim=-1).clamp_min(1e-6)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adj * inv_sqrt[None, :]


class GraphConv(nn.Module):
    """Small dense graph convolution for IEEE 13-bus scale experiments."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, nodes, features]
        support = torch.einsum("ij,btjf->btif", adjacency, x)
        y = self.linear(support)
        y = self.norm(y)
        return self.dropout(self.activation(y))

