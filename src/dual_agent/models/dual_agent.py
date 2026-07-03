from __future__ import annotations

import torch
from torch import nn

from dual_agent.models.forecaster import PVScenarioForecaster
from dual_agent.models.surrogate import OPFSurrogate


class DualAgentSystem(nn.Module):
    """End-to-end system: forecaster feeds a differentiable OPF surrogate."""

    def __init__(self, forecaster: PVScenarioForecaster, surrogate: OPFSurrogate) -> None:
        super().__init__()
        self.forecaster = forecaster
        self.surrogate = surrogate

    def forward(
        self,
        pv_history: torch.Tensor,
        load_forecast: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pv_scenarios = self.forecaster(pv_history, adjacency)
        decision = self.surrogate(pv_scenarios, load_forecast, adjacency)
        decision["pv_scenarios"] = pv_scenarios
        return decision

