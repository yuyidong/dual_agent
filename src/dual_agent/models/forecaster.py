from __future__ import annotations

import torch
from torch import nn

from dual_agent.models.graph import GraphConv, normalize_adjacency


class PVScenarioForecaster(nn.Module):
    """Spatio-temporal PV scenario generator.

    Input:
        history: [batch, history_steps, pv_nodes, pv_feature_dim]
        adjacency: [pv_nodes, pv_nodes]

    Output:
        scenarios: [batch, scenarios, horizon_steps, pv_nodes]
    """

    def __init__(
        self,
        pv_feature_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        scenarios: int,
        graph_layers: int = 2,
        temporal_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.scenarios = scenarios

        layers: list[nn.Module] = []
        in_dim = pv_feature_dim
        for _ in range(graph_layers):
            layers.append(GraphConv(in_dim, hidden_dim, dropout=dropout))
            in_dim = hidden_dim
        self.graph_layers = nn.ModuleList(layers)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, min(8, hidden_dim // 16)),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, scenarios * horizon_steps),
        )

    def forward(self, history: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = normalize_adjacency(adjacency).to(history.device)
        x = history
        for layer in self.graph_layers:
            x = layer(x, adjacency)

        batch, time, nodes, hidden = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch * nodes, time, hidden)
        encoded = self.temporal_encoder(x)
        pooled = encoded[:, -1]
        out = self.head(pooled)
        out = out.view(batch, nodes, self.scenarios, self.horizon_steps)
        out = out.permute(0, 2, 3, 1).contiguous()
        return torch.nn.functional.softplus(out)


def energy_score_loss(pred_scenarios: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Energy score for multivariate scenario forecasts.

    pred_scenarios: [batch, scenarios, horizon, nodes]
    target: [batch, horizon, nodes]
    """
    target = target[:, None, :, :]
    term_1 = torch.linalg.vector_norm(pred_scenarios - target, dim=(-2, -1)).mean(dim=1)
    pairwise = pred_scenarios[:, :, None, :, :] - pred_scenarios[:, None, :, :, :]
    term_2 = torch.linalg.vector_norm(pairwise, dim=(-2, -1)).mean(dim=(1, 2))
    return (term_1 - 0.5 * term_2).mean()

