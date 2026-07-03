from __future__ import annotations

import torch
from torch import nn

from dual_agent.models.graph import GraphConv, normalize_adjacency


class OPFSurrogate(nn.Module):
    """Differentiable neural surrogate for multi-step stochastic OPF.

    The surrogate maps PV/load scenarios and grid context to OPF labels:
    dispatch, expected cost, and network risk. It is intentionally lightweight
    enough to sit in the inner loop of decision-focused forecaster training.
    """

    def __init__(
        self,
        pv_nodes: int,
        load_nodes: int,
        horizon_steps: int,
        scenarios: int,
        load_feature_dim: int,
        hidden_dim: int,
        graph_layers: int = 2,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        battery_count: int = 1,
    ) -> None:
        super().__init__()
        self.pv_nodes = pv_nodes
        self.load_nodes = load_nodes
        self.horizon_steps = horizon_steps
        self.scenarios = scenarios
        self.battery_count = battery_count

        # Per node/time features are scenario mean, scenario std, and load features.
        input_dim = 2 + load_feature_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.graph_layers = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim, dropout=dropout) for _ in range(graph_layers)]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, min(8, hidden_dim // 16)),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)

        self.dispatch_head = nn.Linear(hidden_dim, battery_count)
        self.cost_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.risk_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))

    def forward(
        self,
        pv_scenarios: torch.Tensor,
        load_forecast: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # pv_scenarios: [batch, scenarios, horizon, nodes]
        # load_forecast: [batch, horizon, nodes, load_feature_dim]
        pv_mean = pv_scenarios.mean(dim=1)
        pv_std = pv_scenarios.std(dim=1, unbiased=False)

        if load_forecast.size(2) != pv_mean.size(-1):
            raise ValueError("For the first scaffold, load_forecast nodes must align with pv nodes.")

        x = torch.cat([pv_mean[..., None], pv_std[..., None], load_forecast], dim=-1)
        x = self.input_projection(x)

        adjacency = normalize_adjacency(adjacency).to(x.device)
        for layer in self.graph_layers:
            x = x + layer(x, adjacency)

        batch, horizon, nodes, hidden = x.shape
        pooled_nodes = x.mean(dim=2)
        encoded = self.temporal_encoder(pooled_nodes)

        dispatch = self.dispatch_head(encoded)
        summary = encoded.mean(dim=1)
        cost = torch.nn.functional.softplus(self.cost_head(summary)).squeeze(-1)
        risk = torch.nn.functional.softplus(self.risk_head(summary))
        return {
            "dispatch": dispatch,
            "cost": cost,
            "voltage_risk": risk[:, 0],
            "thermal_risk": risk[:, 1],
        }


def surrogate_supervised_loss(
    prediction: dict[str, torch.Tensor],
    target_dispatch: torch.Tensor,
    target_cost: torch.Tensor,
    target_voltage_risk: torch.Tensor,
    target_thermal_risk: torch.Tensor,
) -> torch.Tensor:
    dispatch_loss = torch.nn.functional.mse_loss(prediction["dispatch"], target_dispatch)
    cost_loss = torch.nn.functional.mse_loss(prediction["cost"], target_cost)
    voltage_loss = torch.nn.functional.mse_loss(prediction["voltage_risk"], target_voltage_risk)
    thermal_loss = torch.nn.functional.mse_loss(prediction["thermal_risk"], target_thermal_risk)
    return dispatch_loss + cost_loss + 0.5 * voltage_loss + 0.5 * thermal_loss

