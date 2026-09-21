from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from dual_agent.config import ExperimentConfig


class ScenarioDataset(Dataset[dict[str, torch.Tensor]]):
    """PV/load scenario dataset shared by forecaster, surrogate, and joint training."""

    def __init__(self, path: str | Path) -> None:
        data = np.load(path)
        self.pv_history = torch.from_numpy(data["pv_history"]).float()
        self.pv_scenarios = (
            torch.from_numpy(data["pv_scenarios"]).float()
            if "pv_scenarios" in data
            else None
        )
        self.pv_target = torch.from_numpy(data["pv_target"]).float()
        self.load_forecast = torch.from_numpy(data["load_forecast"]).float()

    def __len__(self) -> int:
        return self.pv_history.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "pv_history": self.pv_history[index],
            "pv_target": self.pv_target[index],
            "load_forecast": self.load_forecast[index],
        }
        if self.pv_scenarios is not None:
            item["pv_scenarios"] = self.pv_scenarios[index]
        return item


def validate_dataset_matches_config(
    dataset: ScenarioDataset,
    config: ExperimentConfig,
    require_scenarios: bool = False,
) -> None:
    expected = {
        "history_steps": config.problem.history_steps,
        "horizon_steps": config.problem.horizon_steps,
        "pv_feature_dim": config.problem.pv_feature_dim,
        "load_feature_dim": config.problem.load_feature_dim,
    }
    actual = {
        "history_steps": dataset.pv_history.size(1),
        "horizon_steps": dataset.pv_target.size(1),
        "pv_feature_dim": dataset.pv_history.size(-1),
        "load_feature_dim": dataset.load_forecast.size(-1),
    }
    if dataset.pv_scenarios is not None:
        expected["scenarios"] = config.problem.scenarios
        actual["scenarios"] = dataset.pv_scenarios.size(1)
    elif require_scenarios:
        mismatches = ["pv_scenarios: missing from dataset"]
        detail = ", ".join(mismatches)
        raise ValueError(
            "Dataset shape does not match config. "
            f"{detail}. Regenerate the dataset with problem.generate_dataset_scenarios=true, "
            "or provide --forecaster-checkpoint so surrogate training can generate scenarios."
        )
    mismatches = [
        f"{key}: config={expected[key]}, data={actual[key]}"
        for key in expected
        if expected[key] != actual[key]
    ]
    if mismatches:
        detail = ", ".join(mismatches)
        raise ValueError(
            "Dataset shape does not match config. "
            f"{detail}. Regenerate the dataset with scripts/generate_dataset.py using this config, "
            "or use a config that matches the dataset."
        )


def make_complete_graph(nodes: int) -> torch.Tensor:
    adjacency = torch.ones(nodes, nodes)
    adjacency.fill_diagonal_(0.0)
    return adjacency


def make_train_test_loaders(
    dataset: Dataset[dict[str, torch.Tensor]],
    batch_size: int,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    test_size = round(len(dataset) * test_fraction)
    if len(dataset) > 1:
        test_size = min(max(test_size, 1), len(dataset) - 1)
    train_size = len(dataset) - test_size
    if train_size == 0 or test_size == 0:
        raise ValueError("dataset must contain at least two samples for a train/test split")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
