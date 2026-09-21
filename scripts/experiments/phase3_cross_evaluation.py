#!/usr/bin/env python3
"""Cross-evaluation experiment for phase-3 iterative adaptation."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import add_src_to_path

add_src_to_path()

from dual_agent.config import load_config
from dual_agent.models.dual_agent import DualAgentSystem
from dual_agent.models.forecaster import PVScenarioForecaster
from dual_agent.models.surrogate import OPFSurrogate
from dual_agent.opendss.lindistflow_torch import TorchLinDistFlowEvaluator
from dual_agent.training.dataset import (
    ScenarioDataset,
    make_complete_graph,
    make_train_test_loaders,
    validate_dataset_matches_config,
)
from dual_agent.training.loops import (
    current_learning_rate,
    evaluate_forecaster_loss,
    make_warmup_cosine_scheduler,
    train_joint_epoch,
    train_surrogate_epoch,
)


def split_epochs(total_epochs: int, rounds: int) -> list[int]:
    base, remainder = divmod(total_epochs, rounds)
    return [base + (index < remainder) for index in range(rounds)]


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def load_module_checkpoint(module: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    module.load_state_dict(payload)


def build_models(config: Any, nodes: int) -> tuple[PVScenarioForecaster, OPFSurrogate]:
    forecaster = PVScenarioForecaster(
        pv_feature_dim=config.problem.pv_feature_dim,
        hidden_dim=config.model.hidden_dim,
        horizon_steps=config.problem.horizon_steps,
        scenarios=config.problem.scenarios,
        graph_layers=config.model.graph_layers,
        temporal_layers=config.model.temporal_layers,
        dropout=config.model.dropout,
    )
    surrogate = OPFSurrogate(
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
    )
    return forecaster, surrogate


@torch.inference_mode()
def evaluate_pair(
    forecaster: PVScenarioForecaster,
    surrogate: OPFSurrogate,
    evaluator: TorchLinDistFlowEvaluator,
    loader: Any,
    adjacency: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    forecaster.eval()
    surrogate.eval()
    evaluator.eval()
    totals: dict[str, float] = {}
    samples = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        scenarios = forecaster(batch["pv_history"], adjacency)
        decision = surrogate(scenarios, batch["load_forecast"], adjacency)
        result = evaluator(
            decision["dispatch"],
            batch["pv_target"].unsqueeze(1),
            batch["load_forecast"],
        )
        batch_size = batch["pv_history"].size(0)
        samples += batch_size
        for key in (
            "operating_cost",
            "energy_cost",
            "curtailment_cost",
            "degradation",
            "network_violation",
            "voltage_violation",
            "line_flow_violation",
            "terminal_soc_deviation",
        ):
            totals[key] = totals.get(key, 0.0) + float(result[key].mean().cpu()) * batch_size
    return {key: value / samples for key, value in totals.items()}


def save_matrix_csv(
    path: Path,
    raw_cost: np.ndarray,
    normalized_cost: np.ndarray,
    pair_metrics: dict[tuple[int, int], dict[str, float]],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "surrogate_state",
                "forecast_state",
                "operating_cost",
                "normalized_operating_cost",
                "energy_cost",
                "curtailment_cost",
                "degradation",
                "network_violation",
                "voltage_violation",
                "line_flow_violation",
                "terminal_soc_deviation",
            ]
        )
        for i in range(raw_cost.shape[0]):
            for j in range(raw_cost.shape[1]):
                metrics = pair_metrics[(i, j)]
                writer.writerow(
                    [
                        f"S{i}",
                        f"F{j}",
                        f"{raw_cost[i, j]:.10f}",
                        f"{normalized_cost[i, j]:.10f}",
                        f"{metrics['energy_cost']:.10f}",
                        f"{metrics['curtailment_cost']:.10f}",
                        f"{metrics['degradation']:.10f}",
                        f"{metrics['network_violation']:.10f}",
                        f"{metrics['voltage_violation']:.10f}",
                        f"{metrics['line_flow_violation']:.10f}",
                        f"{metrics['terminal_soc_deviation']:.10f}",
                    ]
                )


def plot_cross_matrix(matrix: np.ndarray, output_path: Path) -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    plt.rcParams["axes.unicode_minus"] = False
    size = matrix.shape[0]
    labels = [f"$F_{{{index}}}$" for index in range(size)]
    row_labels = [f"$S_{{{index}}}$" for index in range(size)]
    fig, ax = plt.subplots(figsize=(12.2, 10.2), dpi=180)
    vmin = float(matrix.min())
    vmax = float(matrix.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-3
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(size), labels=labels, fontsize=18)
    ax.set_yticks(np.arange(size), labels=row_labels, fontsize=18)
    ax.set_xlabel("Forecaster state", fontsize=21, labelpad=18)
    ax.set_ylabel("Surrogate state", fontsize=21, labelpad=22)
    ax.tick_params(length=0, pad=8)
    ax.set_title(
        "Iterative Decision Adaptation under Scenario Distribution Shift\nCross-Evaluation Matrix: Normalized Dispatch Cost",
        fontsize=25, fontweight="bold", pad=28,
    )
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    midpoint = (vmin + vmax) / 2.0
    for i in range(size):
        for j in range(size):
            color = "white" if matrix[i, j] >= midpoint else "#102a43"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=17 if size <= 6 else 14, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.08)
    colorbar.set_label("Normalized dispatch cost (lower is better)", fontsize=16, labelpad=15)
    colorbar.ax.tick_params(labelsize=13)
    fig.text(
        0.5, 0.035,
        "All combinations use the (S0, F0) test-set operating cost as a common baseline.",
        ha="center", fontsize=14,
    )
    fig.subplots_adjust(left=0.15, right=0.86, top=0.82, bottom=0.16)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ieee13.yaml")
    parser.add_argument("--data", default="data/ieee13.npz")
    parser.add_argument("--forecaster-checkpoint", default="forecaster.pt")
    parser.add_argument("--surrogate-checkpoint", default="surrogate.pt")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--forecaster-epochs-total", type=int, default=None)
    parser.add_argument("--surrogate-epochs-total", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    rounds = args.rounds or config.training.phase3_rounds
    forecaster_epochs_total = (
        args.forecaster_epochs_total
        if args.forecaster_epochs_total is not None
        else config.training.phase3_forecaster_epochs
    )
    surrogate_epochs_total = (
        args.surrogate_epochs_total
        if args.surrogate_epochs_total is not None
        else config.training.phase3_surrogate_epochs
    )
    if rounds < 1:
        raise ValueError("--rounds must be positive")
    if forecaster_epochs_total < rounds:
        raise ValueError("Total forecaster epochs must be at least the number of rounds.")
    if surrogate_epochs_total < rounds:
        raise ValueError("Total surrogate epochs must be at least the number of rounds.")

    seed = config.seed if args.seed is None else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = ScenarioDataset(args.data)
    validate_dataset_matches_config(dataset, config)
    train_loader, test_loader = make_train_test_loaders(
        dataset, batch_size=config.training.batch_size,
        test_fraction=args.test_fraction, seed=seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes = dataset.pv_history.size(2)
    adjacency = make_complete_graph(nodes).to(device)
    forecaster, surrogate = build_models(config, nodes)
    load_module_checkpoint(forecaster, Path(args.forecaster_checkpoint))
    load_module_checkpoint(surrogate, Path(args.surrogate_checkpoint))
    system = DualAgentSystem(forecaster, surrogate).to(device)
    evaluator = TorchLinDistFlowEvaluator(
        config.network.dss_master, config.network.pv_systems,
        config.network.batteries, config.lindistflow, config.problem.interval_hours,
    ).to(device)

    forecaster_optimizer = torch.optim.AdamW(
        system.forecaster.parameters(), lr=config.training.joint_learning_rate
    )
    surrogate_optimizer = torch.optim.AdamW(
        system.surrogate.parameters(), lr=config.training.joint_surrogate_learning_rate
    )
    initial_forecast_loss = evaluate_forecaster_loss(
        system.forecaster, train_loader, adjacency, device
    )
    forecast_loss_cap = initial_forecast_loss * (
        1.0 + config.training.joint_forecast_loss_tolerance
    )
    forecaster_epochs = split_epochs(forecaster_epochs_total, rounds)
    surrogate_epochs = split_epochs(surrogate_epochs_total, rounds)
    forecaster_scheduler = make_warmup_cosine_scheduler(
        forecaster_optimizer, max(1, sum(forecaster_epochs)), len(train_loader),
        config.training.warmup_epochs, config.training.min_learning_rate_ratio,
    )
    surrogate_scheduler = make_warmup_cosine_scheduler(
        surrogate_optimizer, max(1, sum(surrogate_epochs)), len(train_loader),
        config.training.warmup_epochs, config.training.min_learning_rate_ratio,
    )

    forecaster_states = [cpu_state_dict(forecaster)]
    surrogate_states = [cpu_state_dict(surrogate)]
    round_records: list[dict[str, Any]] = [{
        "stage": 0, "forecaster_epochs": 0, "surrogate_epochs": 0,
        "forecast_loss_cap": forecast_loss_cap,
    }]
    print(f"device={device}")
    print(
        f"dataset={args.data} train={len(train_loader.dataset)} "
        f"test={len(test_loader.dataset)} batch_size={config.training.batch_size}"
    )
    print(
        f"cross_states={rounds + 1} total_forecaster_epochs={forecaster_epochs_total} "
        f"total_surrogate_epochs={surrogate_epochs_total}"
    )
    print(f"initial_train_forecast_loss={initial_forecast_loss:.6f}")

    for round_index in range(rounds):
        round_start = time.time()
        last_surrogate_metrics = None
        last_forecaster_metrics = None
        reference_surrogate = None
        if (
            surrogate_epochs[round_index]
            and config.training.surrogate_consistency_loss_weight > 0.0
        ):
            reference_surrogate = copy.deepcopy(system.surrogate).to(device).eval()
            for parameter in reference_surrogate.parameters():
                parameter.requires_grad_(False)
        for _ in range(surrogate_epochs[round_index]):
            last_surrogate_metrics = train_surrogate_epoch(
                system.surrogate, evaluator, train_loader, surrogate_optimizer,
                adjacency, device, system.forecaster,
                config.training.operating_cost_loss_weight,
                config.training.voltage_violation_loss_weight,
                config.training.line_flow_violation_loss_weight,
                config.training.kw_violation_loss_weight,
                config.training.soc_violation_loss_weight,
                config.training.terminal_soc_loss_weight,
                surrogate_scheduler, config.training.max_grad_norm,
                reference_surrogate,
                config.training.surrogate_consistency_loss_weight,
            )
        if reference_surrogate is not None:
            del reference_surrogate
        for _ in range(forecaster_epochs[round_index]):
            last_forecaster_metrics = train_joint_epoch(
                system, evaluator, train_loader, forecaster_optimizer,
                adjacency, device, forecast_loss_cap,
                config.training.joint_forecast_constraint_weight,
                config.training.operating_cost_loss_weight,
                config.training.voltage_violation_loss_weight,
                config.training.line_flow_violation_loss_weight,
                config.training.kw_violation_loss_weight,
                config.training.soc_violation_loss_weight,
                config.training.terminal_soc_loss_weight,
                forecaster_scheduler, config.training.max_grad_norm,
            )
        forecaster_states.append(cpu_state_dict(forecaster))
        surrogate_states.append(cpu_state_dict(surrogate))
        record: dict[str, Any] = {
            "stage": round_index + 1,
            "forecaster_epochs": forecaster_epochs[round_index],
            "surrogate_epochs": surrogate_epochs[round_index],
            "elapsed_seconds": time.time() - round_start,
            "forecaster_learning_rate": current_learning_rate(forecaster_optimizer),
            "surrogate_learning_rate": current_learning_rate(surrogate_optimizer),
        }
        if last_surrogate_metrics is not None:
            record["surrogate_train_loss"] = last_surrogate_metrics["loss"]
            record["surrogate_train_operating_cost"] = last_surrogate_metrics["operating_cost"]
        if last_forecaster_metrics is not None:
            record["forecaster_train_loss"] = last_forecaster_metrics["loss"]
            record["forecaster_train_forecast_loss"] = last_forecaster_metrics["forecast_loss"]
            record["forecaster_train_operating_cost"] = last_forecaster_metrics["operating_cost"]
        round_records.append(record)
        print(
            f"stage={round_index + 1} surrogate_epochs={surrogate_epochs[round_index]} "
            f"forecaster_epochs={forecaster_epochs[round_index]} "
            f"surrogate_loss={record.get('surrogate_train_loss', float('nan')):.6f} "
            f"forecaster_loss={record.get('forecaster_train_loss', float('nan')):.6f} "
            f"elapsed={record['elapsed_seconds']:.1f}s"
        )

    pair_metrics: dict[tuple[int, int], dict[str, float]] = {}
    raw_cost = np.zeros((rounds + 1, rounds + 1), dtype=np.float64)
    print("evaluating cross combinations...")
    for i, surrogate_state in enumerate(surrogate_states):
        surrogate.load_state_dict(surrogate_state)
        for j, forecaster_state in enumerate(forecaster_states):
            forecaster.load_state_dict(forecaster_state)
            metrics = evaluate_pair(
                forecaster, surrogate, evaluator, test_loader, adjacency, device
            )
            pair_metrics[(i, j)] = metrics
            raw_cost[i, j] = metrics["operating_cost"]
        print(f"S{i}: " + " ".join(f"{raw_cost[i, j]:.3f}" for j in range(rounds + 1)))

    baseline_cost = float(raw_cost[0, 0])
    normalized_cost = raw_cost / baseline_cost
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "phase3_cross_evaluation_matrix.png"
    svg_path = output_dir / "phase3_cross_evaluation_matrix.svg"
    csv_path = output_dir / "phase3_cross_evaluation_matrix.csv"
    json_path = output_dir / "phase3_cross_evaluation_summary.json"
    npz_path = output_dir / "phase3_cross_evaluation_matrix.npz"
    plot_cross_matrix(normalized_cost, matrix_path)
    plot_cross_matrix(normalized_cost, svg_path)
    save_matrix_csv(csv_path, raw_cost, normalized_cost, pair_metrics)

    diagonal = np.diag(normalized_cost)
    fixed_s0_change = normalized_cost[0, -1] - normalized_cost[0, 0]
    matched_change = normalized_cost[-1, -1] - normalized_cost[0, 0]
    summary = {
        "experiment": "phase3_cross_evaluation_iterative_adaptation",
        "config": str(Path(args.config)), "data": str(Path(args.data)),
        "device": str(device), "seed": seed, "test_fraction": args.test_fraction,
        "rounds": rounds,
        "state_labels": {
            "forecast": [f"F{i}" for i in range(rounds + 1)],
            "surrogate": [f"S{i}" for i in range(rounds + 1)],
        },
        "definition": "E[i,j] is the test-set operating cost of surrogate S_i with forecaster F_j; all normalized values divide by E[0,0].",
        "baseline_operating_cost": baseline_cost,
        "raw_operating_cost": raw_cost.tolist(),
        "normalized_operating_cost": normalized_cost.tolist(),
        "diagonal_normalized_cost": diagonal.tolist(),
        "fixed_S0_relative_change_at_last_forecaster": float(fixed_s0_change),
        "matched_pair_relative_change_from_initial": float(matched_change),
        "round_records": round_records,
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "png": str(matrix_path), "svg": str(svg_path),
            "csv": str(csv_path), "npz": str(npz_path),
        },
    }
    np.savez(npz_path, raw_operating_cost=raw_cost, normalized_operating_cost=normalized_cost)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("normalized matrix:")
    for row in normalized_cost:
        print(" ".join(f"{value:.4f}" for value in row))
    print(f"baseline_operating_cost={baseline_cost:.6f}")
    print(f"fixed_S0_last_relative_change={fixed_s0_change:+.4%}")
    print(f"matched_last_relative_change={matched_change:+.4%}")
    print(f"outputs={output_dir.resolve()}")


if __name__ == "__main__":
    main()

