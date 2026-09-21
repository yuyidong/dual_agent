from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

import matplotlib.pyplot as plt
import numpy as np
import torch

from dual_agent.config import load_config
from dual_agent.models.dual_agent import DualAgentSystem
from dual_agent.models.forecaster import PVScenarioForecaster
from dual_agent.models.surrogate import OPFSurrogate
from dual_agent.opendss.lindistflow_torch import TorchLinDistFlowEvaluator
from dual_agent.training.dataset import ScenarioDataset, make_complete_graph, make_train_test_loaders
from dual_agent.training.dataset import validate_dataset_matches_config


@dataclass
class DispatchExample:
    test_position: int
    before_dispatch: np.ndarray
    after_dispatch: np.ndarray
    before_soc: np.ndarray
    after_soc: np.ndarray
    before_pv_mean: np.ndarray
    after_pv_mean: np.ndarray
    before_pv_p10: np.ndarray
    before_pv_p90: np.ndarray
    after_pv_p10: np.ndarray
    after_pv_p90: np.ndarray
    total_pv: np.ndarray
    total_load: np.ndarray
    before_metrics: dict[str, float]
    after_metrics: dict[str, float]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ieee13.yaml")
    parser.add_argument("--data", default="data/ieee13.npz")
    parser.add_argument("--forecaster-checkpoint", default="forecaster.pt")
    parser.add_argument("--surrogate-checkpoint", default="surrogate.pt")
    parser.add_argument("--joint-checkpoint", default="dual_agent.pt")
    parser.add_argument("--out", default="data/battery_dispatch_before_after_phase3.png")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--test-position",
        type=int,
        default=None,
        help="Optional position within the deterministic test split. Defaults to the largest dispatch-change case.",
    )
    parser.add_argument(
        "--selection",
        choices=("dispatch-change", "cost-improvement"),
        default="dispatch-change",
        help="How to select the illustrative test day when --test-position is not provided.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = ScenarioDataset(args.data)
    validate_dataset_matches_config(dataset, config)
    _, test_loader = make_train_test_loaders(
        dataset,
        batch_size=config.training.batch_size,
        test_fraction=args.test_fraction,
        seed=config.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes = dataset.pv_history.size(2)
    adjacency = make_complete_graph(nodes).to(device)

    before_forecaster = _make_forecaster(config, device)
    before_forecaster.load_state_dict(torch.load(args.forecaster_checkpoint, map_location=device))
    before_forecaster.eval()

    before_surrogate = _make_surrogate(config, nodes, device)
    before_surrogate.load_state_dict(torch.load(args.surrogate_checkpoint, map_location=device))
    before_surrogate.eval()

    after_system = DualAgentSystem(_make_forecaster(config, device), _make_surrogate(config, nodes, device)).to(device)
    after_system.load_state_dict(torch.load(args.joint_checkpoint, map_location=device))
    after_system.eval()

    evaluator = TorchLinDistFlowEvaluator(
        config.network.dss_master,
        config.network.pv_systems,
        config.network.batteries,
        config.lindistflow,
        config.problem.interval_hours,
    ).to(device)
    evaluator.eval()

    example = _select_example(
        before_forecaster,
        before_surrogate,
        after_system,
        evaluator,
        test_loader,
        adjacency,
        device,
        config.problem.interval_hours,
        config.network.batteries,
        args.test_position,
        args.selection,
    )
    _plot_example(example, config.problem.interval_hours, Path(args.out))
    print(f"saved_plot={Path(args.out).resolve()}")
    print(f"test_position={example.test_position}")
    print(_format_metrics("before", example.before_metrics))
    print(_format_metrics("after", example.after_metrics))


def _make_forecaster(config, device: torch.device) -> PVScenarioForecaster:
    return PVScenarioForecaster(
        pv_feature_dim=config.problem.pv_feature_dim,
        hidden_dim=config.model.hidden_dim,
        horizon_steps=config.problem.horizon_steps,
        scenarios=config.problem.scenarios,
        graph_layers=config.model.graph_layers,
        temporal_layers=config.model.temporal_layers,
        dropout=config.model.dropout,
    ).to(device)


def _make_surrogate(config, nodes: int, device: torch.device) -> OPFSurrogate:
    return OPFSurrogate(
        pv_nodes=nodes,
        load_nodes=nodes,
        horizon_steps=config.problem.horizon_steps,
        scenarios=config.problem.scenarios,
        load_feature_dim=config.problem.load_feature_dim,
        hidden_dim=config.model.hidden_dim,
        graph_layers=config.model.graph_layers,
        temporal_layers=config.model.temporal_layers,
        dropout=config.model.dropout,
        battery=config.network.batteries,
        interval_hours=config.problem.interval_hours,
    ).to(device)


@torch.no_grad()
def _select_example(
    before_forecaster: PVScenarioForecaster,
    before_surrogate: OPFSurrogate,
    after_system: DualAgentSystem,
    evaluator: TorchLinDistFlowEvaluator,
    test_loader,
    adjacency: torch.Tensor,
    device: torch.device,
    interval_hours: float,
    battery,
    requested_position: int | None,
    selection: str,
) -> DispatchExample:
    best_score = -float("inf")
    best_example: DispatchExample | None = None
    seen = 0
    for batch in test_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        before_scenarios = before_forecaster(batch["pv_history"], adjacency)
        before_output = before_surrogate(before_scenarios, batch["load_forecast"], adjacency)
        after_output = after_system(batch["pv_history"], batch["load_forecast"], adjacency)
        after_scenarios = after_output["pv_scenarios"]

        before_dispatch = before_output["dispatch"]
        after_dispatch = after_output["dispatch"]
        for batch_idx in range(before_dispatch.size(0)):
            position = seen + batch_idx
            if requested_position is not None and position != requested_position:
                continue
            before_metrics = _evaluate_dispatch(evaluator, before_dispatch, batch, batch_idx)
            after_metrics = _evaluate_dispatch(evaluator, after_dispatch, batch, batch_idx)
            score = _selection_score(
                selection,
                before_dispatch[batch_idx],
                after_dispatch[batch_idx],
                before_metrics,
                after_metrics,
            )
            if requested_position is None and score <= best_score:
                continue
            best_score = score
            best_example = DispatchExample(
                test_position=position,
                before_dispatch=before_dispatch[batch_idx].sum(dim=-1).detach().cpu().numpy(),
                after_dispatch=after_dispatch[batch_idx].sum(dim=-1).detach().cpu().numpy(),
                before_soc=_soc_trajectory(before_dispatch[batch_idx], interval_hours, battery)
                .mean(dim=-1)
                .detach()
                .cpu()
                .numpy(),
                after_soc=_soc_trajectory(after_dispatch[batch_idx], interval_hours, battery)
                .mean(dim=-1)
                .detach()
                .cpu()
                .numpy(),
                before_pv_mean=_total_pv_scenario_stat(before_scenarios[batch_idx], "mean"),
                after_pv_mean=_total_pv_scenario_stat(after_scenarios[batch_idx], "mean"),
                before_pv_p10=_total_pv_scenario_stat(before_scenarios[batch_idx], "p10"),
                before_pv_p90=_total_pv_scenario_stat(before_scenarios[batch_idx], "p90"),
                after_pv_p10=_total_pv_scenario_stat(after_scenarios[batch_idx], "p10"),
                after_pv_p90=_total_pv_scenario_stat(after_scenarios[batch_idx], "p90"),
                total_pv=batch["pv_target"][batch_idx].sum(dim=-1).detach().cpu().numpy(),
                total_load=batch["load_forecast"][batch_idx, :, :, 0].sum(dim=-1).detach().cpu().numpy(),
                before_metrics=before_metrics,
                after_metrics=after_metrics,
            )
        seen += before_dispatch.size(0)
    if best_example is None:
        raise ValueError("No dispatch example was selected. Check --test-position and test split size.")
    return best_example


def _selection_score(
    selection: str,
    before_dispatch: torch.Tensor,
    after_dispatch: torch.Tensor,
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
) -> float:
    if selection == "dispatch-change":
        return float((after_dispatch - before_dispatch).abs().mean().detach().cpu())
    if selection == "cost-improvement":
        return before_metrics["operating_cost"] - after_metrics["operating_cost"]
    raise ValueError(f"Unknown selection mode: {selection}")


def _evaluate_dispatch(
    evaluator: TorchLinDistFlowEvaluator,
    dispatch: torch.Tensor,
    batch: dict[str, torch.Tensor],
    batch_idx: int,
) -> dict[str, float]:
    result = evaluator(
        dispatch[batch_idx : batch_idx + 1],
        batch["pv_target"][batch_idx : batch_idx + 1].unsqueeze(1),
        batch["load_forecast"][batch_idx : batch_idx + 1],
    )
    keys = (
        "operating_cost",
        "curtailment_cost",
        "charge_energy_kwh",
        "discharge_energy_kwh",
        "soc_min",
        "soc_max",
        "terminal_soc_deviation",
    )
    return {key: float(result[key].detach().cpu().reshape(-1)[0]) for key in keys}


def _soc_trajectory(dispatch_kw: torch.Tensor, interval_hours: float, battery) -> torch.Tensor:
    batteries = battery if isinstance(battery, tuple) else (battery,)
    kwh_rated = torch.tensor([item.kwh_rated for item in batteries], device=dispatch_kw.device, dtype=dispatch_kw.dtype)
    soc_initial = torch.tensor(
        [item.soc_initial for item in batteries], device=dispatch_kw.device, dtype=dispatch_kw.dtype
    )
    charge_efficiency = torch.tensor(
        [item.charge_efficiency for item in batteries], device=dispatch_kw.device, dtype=dispatch_kw.dtype
    )
    discharge_efficiency = torch.tensor(
        [item.discharge_efficiency for item in batteries], device=dispatch_kw.device, dtype=dispatch_kw.dtype
    )
    if dispatch_kw.ndim == 1:
        dispatch_kw = dispatch_kw.unsqueeze(-1)
    discharge_coeff = interval_hours / (kwh_rated[None, :] * discharge_efficiency[None, :])
    charge_coeff = interval_hours * charge_efficiency[None, :] / kwh_rated[None, :]
    delta = -discharge_coeff * torch.relu(dispatch_kw) + charge_coeff * torch.relu(-dispatch_kw)
    return soc_initial[None, :] + torch.cumsum(delta, dim=0)


def _total_pv_scenario_stat(scenarios: torch.Tensor, stat: str) -> np.ndarray:
    total = scenarios.sum(dim=-1)
    if stat == "mean":
        value = total.mean(dim=0)
    elif stat == "p10":
        value = torch.quantile(total, 0.10, dim=0)
    elif stat == "p90":
        value = torch.quantile(total, 0.90, dim=0)
    else:
        raise ValueError(f"Unknown scenario statistic: {stat}")
    return value.detach().cpu().numpy()


def _plot_example(example: DispatchExample, interval_hours: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    hours = np.arange(example.before_dispatch.shape[0]) * interval_hours

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(hours, example.total_load, label="Total load", color="#454545", linewidth=2.0)
    axes[0].plot(hours, example.total_pv, label="Actual total PV", color="#d99000", linewidth=2.0)
    axes[0].set_ylabel("Power (kW)")
    axes[0].set_title(f"Illustrative Test Day: position {example.test_position}")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].fill_between(
        hours,
        example.before_pv_p10,
        example.before_pv_p90,
        color="#6aaed6",
        alpha=0.24,
        label="Before phase 3 10-90%",
    )
    axes[1].fill_between(
        hours,
        example.after_pv_p10,
        example.after_pv_p90,
        color="#154c9f",
        alpha=0.20,
        label="After phase 3 10-90%",
    )
    axes[1].plot(hours, example.before_pv_mean, label="Before phase 3 mean", color="#6aaed6", linewidth=2.0)
    axes[1].plot(hours, example.after_pv_mean, label="After phase 3 mean", color="#154c9f", linewidth=2.0)
    axes[1].plot(hours, example.total_pv, label="Actual total PV", color="#d99000", linewidth=2.0)
    axes[1].set_ylabel("PV forecast (kW)")
    axes[1].set_title("Total PV forecast scenarios")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].axhline(0.0, color="#777777", linewidth=1.0)
    bar_width = 0.36 * interval_hours
    axes[2].bar(
        hours - bar_width / 2,
        example.before_dispatch,
        width=bar_width,
        label="Before phase 3",
        color="#6aaed6",
        alpha=0.85,
    )
    axes[2].bar(
        hours + bar_width / 2,
        example.after_dispatch,
        width=bar_width,
        label="After phase 3",
        color="#154c9f",
        alpha=0.85,
    )
    axes[2].set_ylabel("Dispatch (kW)")
    axes[2].set_title("Total battery dispatch; positive = discharge, negative = charge")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    axes[3].plot(hours, example.before_soc, label="Before phase 3", color="#6aaed6", linewidth=2.0)
    axes[3].plot(hours, example.after_soc, label="After phase 3", color="#154c9f", linewidth=2.0)
    axes[3].set_xlabel("Hour")
    axes[3].set_ylabel("SOC")
    axes[3].set_title("Average battery SOC")
    axes[3].legend()
    axes[3].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=180)


def _format_metrics(label: str, metrics: dict[str, float]) -> str:
    return (
        f"{label}: operating_cost={metrics['operating_cost']:.3f} "
        f"curtailment_cost={metrics['curtailment_cost']:.3f} "
        f"charge_kwh={metrics['charge_energy_kwh']:.3f} "
        f"discharge_kwh={metrics['discharge_energy_kwh']:.3f} "
        f"soc_min={metrics['soc_min']:.3f} soc_max={metrics['soc_max']:.3f} "
        f"terminal_soc_deviation={metrics['terminal_soc_deviation']:.3f}"
    )


if __name__ == "__main__":
    main()
