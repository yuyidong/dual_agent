from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TeacherDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset schema shared by surrogate and joint training."""

    def __init__(self, path: str | Path) -> None:
        data = np.load(path)
        self.pv_history = torch.from_numpy(data["pv_history"]).float()
        self.pv_scenarios = torch.from_numpy(data["pv_scenarios"]).float()
        self.pv_target = torch.from_numpy(data["pv_target"]).float()
        self.load_forecast = torch.from_numpy(data["load_forecast"]).float()
        self.dispatch = torch.from_numpy(data["dispatch"]).float()
        self.cost = torch.from_numpy(data["cost"]).float()
        self.voltage_risk = torch.from_numpy(data["voltage_risk"]).float()
        self.thermal_risk = torch.from_numpy(data["thermal_risk"]).float()

    def __len__(self) -> int:
        return self.pv_history.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "pv_history": self.pv_history[index],
            "pv_scenarios": self.pv_scenarios[index],
            "pv_target": self.pv_target[index],
            "load_forecast": self.load_forecast[index],
            "dispatch": self.dispatch[index],
            "cost": self.cost[index],
            "voltage_risk": self.voltage_risk[index],
            "thermal_risk": self.thermal_risk[index],
        }


def make_complete_graph(nodes: int) -> torch.Tensor:
    adjacency = torch.ones(nodes, nodes)
    adjacency.fill_diagonal_(0.0)
    return adjacency

