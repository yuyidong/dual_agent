from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from dual_agent.models.dual_agent import DualAgentSystem
from dual_agent.models.forecaster import energy_score_loss
from dual_agent.models.forecaster import PVScenarioForecaster
from dual_agent.models.surrogate import OPFSurrogate
from dual_agent.opendss.lindistflow_torch import TorchLinDistFlowEvaluator


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
    min_learning_rate_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)
    min_ratio = max(0.0, min_learning_rate_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_forecaster_epoch(
    model: PVScenarioForecaster,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    adjacency: torch.Tensor,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_grad_norm: float | None = None,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        prediction = model(batch["pv_history"], adjacency.to(device))
        loss = energy_score_loss(prediction, batch["pv_target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.detach().cpu()) * batch["pv_history"].size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_forecaster_loss(
    model: PVScenarioForecaster,
    loader: DataLoader[dict[str, torch.Tensor]],
    adjacency: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        prediction = model(batch["pv_history"], adjacency.to(device))
        loss = energy_score_loss(prediction, batch["pv_target"])
        total += float(loss.detach().cpu()) * batch["pv_history"].size(0)
    return total / len(loader.dataset)


def train_surrogate_epoch(
    model: OPFSurrogate,
    evaluator: TorchLinDistFlowEvaluator,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    adjacency: torch.Tensor,
    device: torch.device,
    forecaster: PVScenarioForecaster | None,
    cost_weight: float,
    voltage_violation_weight: float,
    line_flow_violation_weight: float,
    kw_violation_weight: float,
    soc_weight: float,
    terminal_soc_weight: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_grad_norm: float | None = None,
    reference_model: OPFSurrogate | None = None,
    consistency_weight: float = 0.0,
) -> dict[str, float]:
    model.train()
    if forecaster is not None:
        forecaster.eval()
    evaluator.eval()
    totals: dict[str, float] = {}
    samples = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        pv_scenarios = _surrogate_training_scenarios(batch, adjacency, device, forecaster)
        prediction = model(pv_scenarios, batch["load_forecast"], adjacency.to(device))
        result = evaluator(
            prediction["dispatch"],
            batch["pv_target"].unsqueeze(1),
            batch["load_forecast"],
        )
        loss = _lindistflow_surrogate_loss(
            result,
            cost_weight,
            voltage_violation_weight,
            line_flow_violation_weight,
            kw_violation_weight,
            soc_weight,
            terminal_soc_weight,
        )
        consistency_penalty = None
        if reference_model is not None and consistency_weight > 0.0:
            with torch.no_grad():
                reference_prediction = reference_model(
                    pv_scenarios, batch["load_forecast"], adjacency.to(device)
                )
            consistency_penalty = consistency_weight * _surrogate_dispatch_consistency(
                model, prediction["dispatch"], reference_prediction["dispatch"], result["operating_cost"]
            )
            loss = loss + consistency_penalty.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        batch_size = batch["pv_history"].size(0)
        samples += batch_size
        _accumulate_surrogate_metrics(
            totals,
            result,
            loss,
            batch_size,
            cost_weight,
            voltage_violation_weight,
            line_flow_violation_weight,
            kw_violation_weight,
            soc_weight,
            terminal_soc_weight,
            consistency_penalty,
        )
    return {key: value / samples for key, value in totals.items()}


@torch.no_grad()
def evaluate_surrogate_loss(
    model: OPFSurrogate,
    evaluator: TorchLinDistFlowEvaluator,
    loader: DataLoader[dict[str, torch.Tensor]],
    adjacency: torch.Tensor,
    device: torch.device,
    forecaster: PVScenarioForecaster | None,
    cost_weight: float,
    voltage_violation_weight: float,
    line_flow_violation_weight: float,
    kw_violation_weight: float,
    soc_weight: float,
    terminal_soc_weight: float,
    reference_model: OPFSurrogate | None = None,
    consistency_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    if forecaster is not None:
        forecaster.eval()
    evaluator.eval()
    totals: dict[str, float] = {}
    samples = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        pv_scenarios = _surrogate_training_scenarios(batch, adjacency, device, forecaster)
        prediction = model(pv_scenarios, batch["load_forecast"], adjacency.to(device))
        result = evaluator(
            prediction["dispatch"],
            batch["pv_target"].unsqueeze(1),
            batch["load_forecast"],
        )
        loss = _lindistflow_surrogate_loss(
            result,
            cost_weight,
            voltage_violation_weight,
            line_flow_violation_weight,
            kw_violation_weight,
            soc_weight,
            terminal_soc_weight,
        )
        consistency_penalty = None
        if reference_model is not None and consistency_weight > 0.0:
            reference_prediction = reference_model(
                pv_scenarios, batch["load_forecast"], adjacency.to(device)
            )
            consistency_penalty = consistency_weight * _surrogate_dispatch_consistency(
                model, prediction["dispatch"], reference_prediction["dispatch"], result["operating_cost"]
            )
            loss = loss + consistency_penalty.mean()
        batch_size = batch["pv_history"].size(0)
        samples += batch_size
        _accumulate_surrogate_metrics(
            totals,
            result,
            loss,
            batch_size,
            cost_weight,
            voltage_violation_weight,
            line_flow_violation_weight,
            kw_violation_weight,
            soc_weight,
            terminal_soc_weight,
            consistency_penalty,
        )
    return {key: value / samples for key, value in totals.items()}


def _surrogate_dispatch_consistency(
    model: OPFSurrogate,
    dispatch: torch.Tensor,
    reference_dispatch: torch.Tensor,
    operating_cost: torch.Tensor,
) -> torch.Tensor:
    if model.kw_rated is not None:
        scale = model.kw_rated.detach().mean().clamp_min(1.0)
    else:
        scale = dispatch.detach().abs().mean().clamp_min(1.0)
    normalized_delta = ((dispatch - reference_dispatch) / scale).pow(2).mean(dim=(1, 2))
    return operating_cost.detach().clamp_min(1.0) * normalized_delta


def _surrogate_training_scenarios(
    batch: dict[str, torch.Tensor],
    adjacency: torch.Tensor,
    device: torch.device,
    forecaster: PVScenarioForecaster | None,
) -> torch.Tensor:
    if forecaster is None:
        if "pv_scenarios" not in batch:
            raise ValueError(
                "Dataset batch does not contain pv_scenarios. "
                "Provide a forecaster checkpoint for surrogate training or regenerate the dataset "
                "with problem.generate_dataset_scenarios=true."
            )
        return batch["pv_scenarios"]
    with torch.no_grad():
        return forecaster(batch["pv_history"], adjacency.to(device))


def _accumulate_surrogate_metrics(
    totals: dict[str, float],
    result: dict[str, torch.Tensor],
    loss: torch.Tensor,
    batch_size: int,
    cost_weight: float,
    voltage_violation_weight: float,
    line_flow_violation_weight: float,
    kw_violation_weight: float,
    soc_weight: float,
    terminal_soc_weight: float,
    consistency_penalty: torch.Tensor | None = None,
) -> None:
    totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu()) * batch_size
    if consistency_penalty is not None:
        totals["surrogate_consistency_loss"] = (
            totals.get("surrogate_consistency_loss", 0.0)
            + float(consistency_penalty.detach().mean().cpu()) * batch_size
        )
    loss_components = _lindistflow_loss_components(
        result,
        cost_weight,
        voltage_violation_weight,
        line_flow_violation_weight,
        kw_violation_weight,
        soc_weight,
        terminal_soc_weight,
    )
    for key, value in loss_components.items():
        totals[key] = totals.get(key, 0.0) + float(value.detach().mean().cpu()) * batch_size
    reconstructed_loss = sum(loss_components.values()).mean()
    reconstruction_error = (reconstructed_loss - loss).abs()
    totals["loss_reconstruction_error"] = (
        totals.get("loss_reconstruction_error", 0.0)
        + float(reconstruction_error.detach().cpu()) * batch_size
    )
    for key, value in result.items():
        totals[key] = totals.get(key, 0.0) + float(value.detach().mean().cpu()) * batch_size


def _lindistflow_surrogate_loss(
    result: dict[str, torch.Tensor],
    cost_weight: float,
    voltage_violation_weight: float,
    line_flow_violation_weight: float,
    kw_violation_weight: float,
    soc_weight: float,
    terminal_soc_weight: float,
) -> torch.Tensor:
    return sum(
        _lindistflow_loss_components(
            result,
            cost_weight,
            voltage_violation_weight,
            line_flow_violation_weight,
            kw_violation_weight,
            soc_weight,
            terminal_soc_weight,
        ).values()
    ).mean()


def _lindistflow_loss_components(
    result: dict[str, torch.Tensor],
    cost_weight: float,
    voltage_violation_weight: float,
    line_flow_violation_weight: float,
    kw_violation_weight: float,
    soc_weight: float,
    terminal_soc_weight: float,
) -> dict[str, torch.Tensor]:
    voltage_violation_loss = voltage_violation_weight * result["voltage_violation"]
    line_flow_violation_loss = line_flow_violation_weight * result["line_flow_violation"]
    kw_violation_loss = kw_violation_weight * result["kw_violation"]
    return {
        "cost_loss": cost_weight * result["operating_cost"],
        "voltage_violation_loss": voltage_violation_loss,
        "line_flow_violation_loss": line_flow_violation_loss,
        "kw_violation_loss": kw_violation_loss,
        "soc_violation_loss": soc_weight * result["soc_violation"],
        "terminal_soc_loss": terminal_soc_weight * result["terminal_soc_deviation"],
    }


def train_joint_epoch(
    system: DualAgentSystem,
    evaluator: TorchLinDistFlowEvaluator,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    adjacency: torch.Tensor,
    device: torch.device,
    forecast_loss_cap: float,
    forecast_constraint_weight: float,
    forecast_loss_weight: float,
    operating_cost_loss_weight: float,
    voltage_violation_loss_weight: float,
    line_flow_violation_loss_weight: float,
    kw_violation_loss_weight: float,
    soc_violation_loss_weight: float,
    terminal_soc_loss_weight: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    system.train()
    system.surrogate.eval()
    evaluator.eval()
    for param in system.surrogate.parameters():
        param.requires_grad_(False)

    totals: dict[str, float] = {}
    samples = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = system(batch["pv_history"], batch["load_forecast"], adjacency.to(device))
        forecast_loss = energy_score_loss(output["pv_scenarios"], batch["pv_target"])
        result = evaluator(
            output["dispatch"],
            batch["pv_target"].unsqueeze(1),
            batch["load_forecast"],
        )
        lindistflow_loss = _lindistflow_surrogate_loss(
            result,
            operating_cost_loss_weight,
            voltage_violation_loss_weight,
            line_flow_violation_loss_weight,
            kw_violation_loss_weight,
            soc_violation_loss_weight,
            terminal_soc_loss_weight,
        )
        forecast_constraint_violation = torch.relu(forecast_loss - forecast_loss_cap)
        loss = (
            lindistflow_loss
            + forecast_loss_weight * forecast_loss
            + forecast_constraint_weight * forecast_constraint_violation
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(system.forecaster.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        batch_size = batch["pv_history"].size(0)
        samples += batch_size
        _accumulate_joint_metrics(
            totals,
            result,
            loss,
            forecast_loss,
            lindistflow_loss,
            forecast_constraint_violation,
            forecast_loss_cap,
            batch_size,
        )

    for param in system.surrogate.parameters():
        param.requires_grad_(True)
    return {key: value / samples for key, value in totals.items()}


@torch.no_grad()
def evaluate_joint_loss(
    system: DualAgentSystem,
    evaluator: TorchLinDistFlowEvaluator,
    loader: DataLoader[dict[str, torch.Tensor]],
    adjacency: torch.Tensor,
    device: torch.device,
    forecast_loss_cap: float,
    forecast_constraint_weight: float,
    forecast_loss_weight: float,
    operating_cost_loss_weight: float,
    voltage_violation_loss_weight: float,
    line_flow_violation_loss_weight: float,
    kw_violation_loss_weight: float,
    soc_violation_loss_weight: float,
    terminal_soc_loss_weight: float,
) -> dict[str, float]:
    system.eval()
    evaluator.eval()
    totals: dict[str, float] = {}
    samples = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = system(batch["pv_history"], batch["load_forecast"], adjacency.to(device))
        forecast_loss = energy_score_loss(output["pv_scenarios"], batch["pv_target"])
        result = evaluator(
            output["dispatch"],
            batch["pv_target"].unsqueeze(1),
            batch["load_forecast"],
        )
        lindistflow_loss = _lindistflow_surrogate_loss(
            result,
            operating_cost_loss_weight,
            voltage_violation_loss_weight,
            line_flow_violation_loss_weight,
            kw_violation_loss_weight,
            soc_violation_loss_weight,
            terminal_soc_loss_weight,
        )
        forecast_constraint_violation = torch.relu(forecast_loss - forecast_loss_cap)
        loss = (
            lindistflow_loss
            + forecast_loss_weight * forecast_loss
            + forecast_constraint_weight * forecast_constraint_violation
        )
        batch_size = batch["pv_history"].size(0)
        samples += batch_size
        _accumulate_joint_metrics(
            totals,
            result,
            loss,
            forecast_loss,
            lindistflow_loss,
            forecast_constraint_violation,
            forecast_loss_cap,
            batch_size,
        )
    return {key: value / samples for key, value in totals.items()}


def _accumulate_joint_metrics(
    totals: dict[str, float],
    result: dict[str, torch.Tensor],
    loss: torch.Tensor,
    forecast_loss: torch.Tensor,
    lindistflow_loss: torch.Tensor,
    forecast_constraint_violation: torch.Tensor,
    forecast_loss_cap: float,
    batch_size: int,
) -> None:
    totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu()) * batch_size
    totals["forecast_loss"] = (
        totals.get("forecast_loss", 0.0) + float(forecast_loss.detach().cpu()) * batch_size
    )
    totals["lindistflow_loss"] = (
        totals.get("lindistflow_loss", 0.0) + float(lindistflow_loss.detach().cpu()) * batch_size
    )
    totals["forecast_constraint_violation"] = (
        totals.get("forecast_constraint_violation", 0.0)
        + float(forecast_constraint_violation.detach().cpu()) * batch_size
    )
    totals["forecast_loss_cap"] = totals.get("forecast_loss_cap", 0.0) + forecast_loss_cap * batch_size
    for key, value in result.items():
        totals[key] = totals.get(key, 0.0) + float(value.detach().mean().cpu()) * batch_size
