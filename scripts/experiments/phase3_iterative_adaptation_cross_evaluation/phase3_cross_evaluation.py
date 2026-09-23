#!/usr/bin/env python3
"""Cross-evaluation experiment for phase-3 iterative adaptation."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _bootstrap import add_src_to_path

add_src_to_path()

from dual_agent.config import load_config
from dual_agent.models.forecaster import PVScenarioForecaster
from dual_agent.models.surrogate import OPFSurrogate
from dual_agent.opendss.lindistflow_torch import TorchLinDistFlowEvaluator
from dual_agent.training.dataset import (
    ScenarioDataset,
    make_complete_graph,
    make_train_validation_test_loaders,
    validate_dataset_matches_config,
)
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


@torch.inference_mode()
def evaluate_forecaster_output_mmd(
    forecaster: PVScenarioForecaster,
    forecaster_states: list[dict[str, torch.Tensor]],
    loader: Any,
    adjacency: torch.Tensor,
    device: torch.device,
    reference: str = "adjacent",
) -> list[float]:
    """Measure MMD across forecaster states.

    ``initial`` reports MMD(F0, Fi); ``adjacent`` reports
    MMD(F{i-1}, Fi), which directly measures successive-stage shift.
    """
    if reference not in {"initial", "adjacent"}:
        raise ValueError("reference must be 'initial' or 'adjacent'")
    def collect_outputs() -> torch.Tensor:
        forecaster.eval()
        chunks = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = forecaster(batch["pv_history"], adjacency)
            chunks.append(output.reshape(output.shape[0], -1).cpu())
        return torch.cat(chunks, dim=0)

    def rbf_mmd(reference: torch.Tensor, current: torch.Tensor) -> float:
        count = min(reference.shape[0], current.shape[0], 256)
        reference = reference[:count].float()
        current = current[:count].float()
        merged = torch.cat([reference, current], dim=0)
        distances = torch.cdist(merged, merged).pow(2)
        positive = distances[distances > 0]
        bandwidth = torch.median(positive).clamp_min(1e-6)
        kernel = torch.exp(-distances / (2.0 * bandwidth))
        xx = kernel[:count, :count]
        yy = kernel[count:, count:]
        xy = kernel[:count, count:]
        return float((xx.mean() + yy.mean() - 2.0 * xy.mean()).clamp_min(0.0))

    outputs = []
    for state in forecaster_states:
        forecaster.load_state_dict(state)
        outputs.append(collect_outputs())
    if reference == "initial":
        initial = outputs[0]
        return [0.0 if index == 0 else rbf_mmd(initial, output)
                for index, output in enumerate(outputs)]
    return [0.0] + [rbf_mmd(outputs[index - 1], outputs[index])
                     for index in range(1, len(outputs))]


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


def plot_cross_matrix(
    matrix: np.ndarray,
    output_path: Path,
    forecaster_mmd: list[float] | None = None,
    mmd_reference: str = "initial",
) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    size = matrix.shape[0]
    if forecaster_mmd is None:
        labels = [f"$F_{{{index}}}$" for index in range(size)]
    elif mmd_reference == "adjacent":
        labels = [f"$F_{{{index}}}$" if index == 0 else
                  f"$F_{{{index}}}$\n$D={forecaster_mmd[index]:.2f}$"
                  for index in range(size)]
    else:
        labels = [f"$F_{{{index}}}$\n$D={forecaster_mmd[index]:.2f}$"
                  for index in range(size)]
    row_labels = [f"$S_{{{index}}}$" for index in range(size)]
    fig, ax = plt.subplots(figsize=(9.8, 8.0), dpi=220)
    vmin = float(matrix.min())
    vmax = float(matrix.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-3
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(
        np.arange(size),
        labels=labels,
        fontsize=12 if mmd_reference == "adjacent" else 15,
    )
    ax.set_yticks(np.arange(size), labels=row_labels, fontsize=16)
    ax.tick_params(axis="x", labelcolor="#2b6cb0")
    if mmd_reference == "adjacent":
        xlabel = r"Forecaster state ($D$: adjacent MMD)"
    else:
        xlabel = "Forecaster state (D: MMD from F0)"
    ax.set_xlabel(xlabel, fontsize=15, labelpad=18)
    ax.set_ylabel("Surrogate state", fontsize=18, labelpad=18)
    ax.tick_params(length=0, pad=8)
    ax.set_title(
        "Cross-evaluation of iterative adaptation",
        fontsize=21, fontweight="bold", pad=22,
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
    for i in range(size):
        ax.add_patch(
            plt.Rectangle(
                (i - 0.43, i - 0.43), 0.86, 0.86,
                fill=False, edgecolor="#202020", linewidth=1.3, zorder=4,
            )
        )
    for i in range(1, size):
        ax.annotate(
            "", xy=(i, i - 0.27), xytext=(i, i - 1 + 0.27),
            arrowprops={"arrowstyle": "->", "color": "#202020", "lw": 1.4},
            zorder=5,
        )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.08)
    colorbar.set_label("Normalized dispatch cost", fontsize=14, labelpad=12)
    colorbar.ax.tick_params(labelsize=11)
    fig.subplots_adjust(left=0.15, right=0.86, top=0.84, bottom=0.22)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ieee13.yaml")
    parser.add_argument("--data", default="data/ieee13/ieee13.npz")
    parser.add_argument(
        "--phase3-checkpoint-dir",
        default="checkpoints/phase3_states",
        help="Directory containing forecaster_XX.pt and surrogate_XX.pt from train_joint.py.",
    )
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    rounds = args.rounds or config.training.phase3_rounds
    if rounds < 1:
        raise ValueError("--rounds must be positive")

    seed = config.seed if args.seed is None else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = ScenarioDataset(args.data)
    validate_dataset_matches_config(dataset, config)
    train_fraction = (
        config.data_split.train_fraction
        if args.train_fraction is None else args.train_fraction
    )
    validation_fraction = (
        config.data_split.validation_fraction
        if args.validation_fraction is None else args.validation_fraction
    )
    test_fraction = (
        config.data_split.test_fraction
        if args.test_fraction is None else args.test_fraction
    )
    train_loader, validation_loader, test_loader = make_train_validation_test_loaders(
        dataset,
        batch_size=config.training.batch_size,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes = dataset.pv_history.size(2)
    adjacency = make_complete_graph(nodes).to(device)
    forecaster, surrogate = build_models(config, nodes)
    forecaster.to(device)
    surrogate.to(device)
    evaluator = TorchLinDistFlowEvaluator(
        config.network.dss_master, config.network.pv_systems,
        config.network.batteries, config.lindistflow, config.problem.interval_hours,
    ).to(device)

    checkpoint_dir = Path(args.phase3_checkpoint_dir)
    forecaster_states: list[dict[str, torch.Tensor]] = []
    surrogate_states: list[dict[str, torch.Tensor]] = []
    for stage in range(rounds + 1):
        forecaster_path = checkpoint_dir / f"forecaster_{stage:02d}.pt"
        surrogate_path = checkpoint_dir / f"surrogate_{stage:02d}.pt"
        if not forecaster_path.is_file() or not surrogate_path.is_file():
            missing = [
                str(path) for path in (forecaster_path, surrogate_path)
                if not path.is_file()
            ]
            raise FileNotFoundError(
                "Missing phase-3 state checkpoint(s): " + ", ".join(missing)
                + ". Run train_joint.py once with --phase3-checkpoint-dir first."
            )
        load_module_checkpoint(forecaster, forecaster_path)
        load_module_checkpoint(surrogate, surrogate_path)
        forecaster_states.append(cpu_state_dict(forecaster))
        surrogate_states.append(cpu_state_dict(surrogate))

    print(f"device={device}")
    print(
        f"dataset={args.data} "
        f"train={len(train_loader.dataset)} "
        f"validation={len(validation_loader.dataset)} "
        f"test={len(test_loader.dataset)} batch_size={config.training.batch_size}"
    )
    print(f"cross_states={rounds + 1}")
    print(f"phase3_checkpoint_dir={checkpoint_dir.resolve()}")
    print("loaded=" + " ".join(
        f"F{i}/S{i}" for i in range(rounds + 1)
    ))

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

    forecaster_mmd = evaluate_forecaster_output_mmd(
        forecaster,
        forecaster_states,
        test_loader,
        adjacency,
        device,
        reference="adjacent",
    )
    print("forecaster_output_mmd_adjacent=" + " ".join(f"{value:.6f}" for value in forecaster_mmd))
    baseline_cost = float(raw_cost[0, 0])
    normalized_cost = raw_cost / baseline_cost
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "phase3_cross_evaluation_matrix.png"
    svg_path = output_dir / "phase3_cross_evaluation_matrix.svg"
    csv_path = output_dir / "phase3_cross_evaluation_matrix.csv"
    json_path = output_dir / "phase3_cross_evaluation_summary.json"
    npz_path = output_dir / "phase3_cross_evaluation_matrix.npz"
    plot_cross_matrix(normalized_cost, matrix_path, forecaster_mmd, mmd_reference="adjacent")
    plot_cross_matrix(normalized_cost, svg_path, forecaster_mmd, mmd_reference="adjacent")
    save_matrix_csv(csv_path, raw_cost, normalized_cost, pair_metrics)

    diagonal = np.diag(normalized_cost)
    fixed_s0_change = normalized_cost[0, -1] - normalized_cost[0, 0]
    matched_change = normalized_cost[-1, -1] - normalized_cost[0, 0]
    summary = {
        "experiment": "phase3_cross_evaluation_iterative_adaptation",
        "config": str(Path(args.config)), "data": str(Path(args.data)),
        "phase3_checkpoint_dir": str(checkpoint_dir),
        "state_source": "saved_phase3_stage_checkpoints",
        "device": str(device), "seed": seed,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
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
        "forecaster_output_mmd_adjacent": forecaster_mmd,
        "fixed_S0_relative_change_at_last_forecaster": float(fixed_s0_change),
        "matched_pair_relative_change_from_initial": float(matched_change),
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
