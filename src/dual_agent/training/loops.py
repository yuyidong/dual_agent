from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from dual_agent.models.dual_agent import DualAgentSystem
from dual_agent.models.forecaster import energy_score_loss
from dual_agent.models.forecaster import PVScenarioForecaster
from dual_agent.models.surrogate import OPFSurrogate, surrogate_supervised_loss


def train_forecaster_epoch(
    model: PVScenarioForecaster,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    adjacency: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        prediction = model(batch["pv_history"], adjacency.to(device))
        loss = energy_score_loss(prediction, batch["pv_target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * batch["pv_history"].size(0)
    return total / len(loader.dataset)


def train_surrogate_epoch(
    model: OPFSurrogate,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    adjacency: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        prediction = model(batch["pv_scenarios"], batch["load_forecast"], adjacency.to(device))
        loss = surrogate_supervised_loss(
            prediction,
            batch["dispatch"],
            batch["cost"],
            batch["voltage_risk"],
            batch["thermal_risk"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * batch["pv_scenarios"].size(0)
    return total / len(loader.dataset)


def train_joint_epoch(
    system: DualAgentSystem,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    adjacency: torch.Tensor,
    device: torch.device,
    forecast_loss_weight: float,
    decision_loss_weight: float,
    risk_loss_weight: float,
) -> float:
    system.train()
    system.surrogate.eval()
    for param in system.surrogate.parameters():
        param.requires_grad_(False)

    total = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = system(batch["pv_history"], batch["load_forecast"], adjacency.to(device))
        forecast_loss = energy_score_loss(output["pv_scenarios"], batch["pv_target"])
        decision_loss = output["cost"].mean()
        risk_loss = output["voltage_risk"].mean() + output["thermal_risk"].mean()
        loss = (
            forecast_loss_weight * forecast_loss
            + decision_loss_weight * decision_loss
            + risk_loss_weight * risk_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * batch["pv_history"].size(0)

    for param in system.surrogate.parameters():
        param.requires_grad_(True)
    return total / len(loader.dataset)
