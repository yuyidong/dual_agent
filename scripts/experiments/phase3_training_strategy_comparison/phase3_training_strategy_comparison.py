#!/usr/bin/env python3
"""Compare phase-3 update strategies at matched alternating rounds."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    make_train_validation_test_loaders,
    validate_dataset_matches_config,
)
from dual_agent.training.loops import (
    disable_training_dropout,
    evaluate_forecaster_loss,
    evaluate_joint_loss,
    evaluate_surrogate_loss,
    make_warmup_cosine_scheduler,
    train_joint_epoch,
    train_surrogate_epoch,
)


# Validation is expensive because it includes the differentiable power-flow
# evaluator. Check regularly while keeping the training budget unchanged.
VALIDATION_INTERVAL = 10


def load_module_checkpoint(module: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    module.load_state_dict(payload)


def discover_phase3_rounds(checkpoint_dir: Path) -> int:
    def collect_stages(prefix: str) -> set[int]:
        stages = set()
        for path in checkpoint_dir.glob(f"{prefix}_*.pt"):
            suffix = path.stem.rsplit("_", 1)[-1]
            if suffix.isdigit():
                stages.add(int(suffix))
        return stages

    forecaster_stages = collect_stages("forecaster")
    surrogate_stages = collect_stages("surrogate")
    if not forecaster_stages or not surrogate_stages:
        raise FileNotFoundError(
            f"No phase-3 stage checkpoints found in {checkpoint_dir}."
        )
    if forecaster_stages != surrogate_stages:
        raise ValueError(
            "Forecaster and surrogate stages do not match: "
            f"forecaster={sorted(forecaster_stages)}, "
            f"surrogate={sorted(surrogate_stages)}"
        )
    last_stage = max(forecaster_stages)
    expected = set(range(last_stage + 1))
    if forecaster_stages != expected:
        missing = sorted(expected - forecaster_stages)
        raise FileNotFoundError(
            f"Missing contiguous phase-3 stage checkpoint(s): {missing}"
        )
    if last_stage < 1:
        raise ValueError("At least one completed phase-3 round is required.")
    return last_stage


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


def build_system(
    config: Any,
    nodes: int,
    device: torch.device,
    disable_dropout: bool = True,
) -> DualAgentSystem:
    forecaster, surrogate = build_models(config, nodes)
    system = DualAgentSystem(forecaster, surrogate).to(device)
    if disable_dropout:
        # Match the deterministic phase-3 behavior used by train_joint.py.
        disable_training_dropout(system)
    return system


def split_epochs(total_epochs: int, rounds: int) -> list[int]:
    base, remainder = divmod(total_epochs, rounds)
    return [base + int(index < remainder) for index in range(rounds)]


def make_evaluator(config: Any, device: torch.device) -> TorchLinDistFlowEvaluator:
    return TorchLinDistFlowEvaluator(
        config.network.dss_master,
        config.network.pv_systems,
        config.network.batteries,
        config.lindistflow,
        config.problem.interval_hours,
    ).to(device)


def joint_validation_objective(
    system: DualAgentSystem,
    evaluator: TorchLinDistFlowEvaluator,
    loader: Any,
    adjacency: torch.Tensor,
    device: torch.device,
    forecast_loss_cap: float,
    config: Any,
) -> float:
    metrics = evaluate_joint_loss(
        system,
        evaluator,
        loader,
        adjacency,
        device,
        forecast_loss_cap,
        config.training.joint_forecast_constraint_weight,
        config.training.joint_forecast_loss_weight,
        config.training.operating_cost_loss_weight,
        config.training.voltage_violation_loss_weight,
        config.training.line_flow_violation_loss_weight,
        config.training.kw_violation_loss_weight,
        config.training.soc_violation_loss_weight,
        config.training.terminal_soc_loss_weight,
    )
    return float(metrics["lindistflow_loss"])


def surrogate_validation_objective(
    system: DualAgentSystem,
    evaluator: TorchLinDistFlowEvaluator,
    loader: Any,
    adjacency: torch.Tensor,
    device: torch.device,
    config: Any,
) -> float:
    metrics = evaluate_surrogate_loss(
        system.surrogate,
        evaluator,
        loader,
        adjacency,
        device,
        system.forecaster,
        config.training.operating_cost_loss_weight,
        config.training.voltage_violation_loss_weight,
        config.training.line_flow_violation_loss_weight,
        config.training.kw_violation_loss_weight,
        config.training.soc_violation_loss_weight,
        config.training.terminal_soc_loss_weight,
    )
    return float(metrics["loss"])


def train_forecast_only(
    config: Any,
    system: DualAgentSystem,
    evaluator: TorchLinDistFlowEvaluator,
    train_loader: Any,
    validation_loader: Any,
    test_loader: Any,
    adjacency: torch.Tensor,
    device: torch.device,
    rounds: int,
    checkpoint_dir: Path,
) -> np.ndarray:
    """Extend the phase-3 forecaster update with S0 fixed."""
    for parameter in system.surrogate.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        system.forecaster.parameters(),
        lr=config.training.phase3_forecaster_learning_rate,
    )
    # Match the total per-round update budget of alternating training:
    # one forecaster block plus one surrogate block.
    total_round_epochs = (
        config.training.phase3_forecaster_epochs
        + config.training.phase3_surrogate_epochs
    )
    epoch_budgets = split_epochs(total_round_epochs, rounds)
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        max(1, sum(epoch_budgets)),
        len(train_loader),
        config.training.warmup_epochs,
        config.training.min_learning_rate_ratio,
    )
    initial_forecast_loss = evaluate_forecaster_loss(
        system.forecaster, train_loader, adjacency, device
    )
    forecast_loss_cap = initial_forecast_loss * (
        1.0 + config.training.joint_forecast_loss_tolerance
    )
    costs: list[float] = []
    for round_index, epochs in enumerate(epoch_budgets, start=1):
        best_score = joint_validation_objective(
            system,
            evaluator,
            validation_loader,
            adjacency,
            device,
            forecast_loss_cap,
            config,
        )
        best_state = copy.deepcopy(system.forecaster.state_dict())
        for epoch_index in range(epochs):
            train_joint_epoch(
                system,
                evaluator,
                train_loader,
                optimizer,
                adjacency,
                device,
                forecast_loss_cap,
                config.training.joint_forecast_constraint_weight,
                config.training.joint_forecast_loss_weight,
                config.training.operating_cost_loss_weight,
                config.training.voltage_violation_loss_weight,
                config.training.line_flow_violation_loss_weight,
                config.training.kw_violation_loss_weight,
                config.training.soc_violation_loss_weight,
                config.training.terminal_soc_loss_weight,
                scheduler,
                config.training.max_grad_norm,
            )
            if (
                (epoch_index + 1) % VALIDATION_INTERVAL != 0
                and epoch_index + 1 != epochs
            ):
                continue
            validation_score = joint_validation_objective(
                system,
                evaluator,
                validation_loader,
                adjacency,
                device,
                forecast_loss_cap,
                config,
            )
            if validation_score < best_score:
                best_score = validation_score
                best_state = copy.deepcopy(system.forecaster.state_dict())
        system.forecaster.load_state_dict(best_state)
        torch.save(
            {key: value.detach().cpu().clone() for key, value in best_state.items()},
            checkpoint_dir / f"forecaster_only_{round_index:02d}.pt",
        )
        costs.append(
            evaluate_operating_cost(
                system.forecaster,
                system.surrogate,
                evaluator,
                test_loader,
                adjacency,
                device,
            )
        )
        print(
            f"forecast-only round={round_index:02d}/{rounds:02d} "
            f"epochs={epochs} validation_best={best_score:.6f} "
            f"test_operating_cost={costs[-1]:.6f}"
        )
    return np.asarray(costs, dtype=np.float64)


def train_surrogate_only(
    config: Any,
    system: DualAgentSystem,
    evaluator: TorchLinDistFlowEvaluator,
    train_loader: Any,
    validation_loader: Any,
    test_loader: Any,
    adjacency: torch.Tensor,
    device: torch.device,
    rounds: int,
    checkpoint_dir: Path,
) -> np.ndarray:
    """Extend the phase-2 surrogate update with F0 fixed."""
    for parameter in system.forecaster.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        system.surrogate.parameters(),
        lr=config.training.surrogate_learning_rate,
    )
    # Match the total per-round update budget of alternating training:
    # one forecaster block plus one surrogate block.
    total_round_epochs = (
        config.training.phase3_forecaster_epochs
        + config.training.phase3_surrogate_epochs
    )
    epoch_budgets = split_epochs(total_round_epochs, rounds)
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        max(1, sum(epoch_budgets)),
        len(train_loader),
        config.training.warmup_epochs,
        config.training.min_learning_rate_ratio,
    )
    costs: list[float] = []
    for round_index, epochs in enumerate(epoch_budgets, start=1):
        best_score = surrogate_validation_objective(
            system,
            evaluator,
            validation_loader,
            adjacency,
            device,
            config,
        )
        best_state = copy.deepcopy(system.surrogate.state_dict())
        for epoch_index in range(epochs):
            train_surrogate_epoch(
                system.surrogate,
                evaluator,
                train_loader,
                optimizer,
                adjacency,
                device,
                system.forecaster,
                config.training.operating_cost_loss_weight,
                config.training.voltage_violation_loss_weight,
                config.training.line_flow_violation_loss_weight,
                config.training.kw_violation_loss_weight,
                config.training.soc_violation_loss_weight,
                config.training.terminal_soc_loss_weight,
                scheduler,
                config.training.max_grad_norm,
            )
            if (
                (epoch_index + 1) % VALIDATION_INTERVAL != 0
                and epoch_index + 1 != epochs
            ):
                continue
            validation_score = surrogate_validation_objective(
                system,
                evaluator,
                validation_loader,
                adjacency,
                device,
                config,
            )
            if validation_score < best_score:
                best_score = validation_score
                best_state = copy.deepcopy(system.surrogate.state_dict())
        system.surrogate.load_state_dict(best_state)
        torch.save(
            {key: value.detach().cpu().clone() for key, value in best_state.items()},
            checkpoint_dir / f"surrogate_only_{round_index:02d}.pt",
        )
        costs.append(
            evaluate_operating_cost(
                system.forecaster,
                system.surrogate,
                evaluator,
                test_loader,
                adjacency,
                device,
            )
        )
        print(
            f"surrogate-only round={round_index:02d}/{rounds:02d} "
            f"epochs={epochs} validation_best={best_score:.6f} "
            f"test_operating_cost={costs[-1]:.6f}"
        )
    return np.asarray(costs, dtype=np.float64)


@torch.inference_mode()
def evaluate_operating_cost(
    forecaster: PVScenarioForecaster,
    surrogate: OPFSurrogate,
    evaluator: TorchLinDistFlowEvaluator,
    loader: Any,
    adjacency: torch.Tensor,
    device: torch.device,
) -> float:
    forecaster.eval()
    surrogate.eval()
    evaluator.eval()
    total = 0.0
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
        total += float(result["operating_cost"].mean().cpu()) * batch_size
        samples += batch_size
    return total / samples


def evaluate_saved_single_strategy(
    strategy: str,
    system: DualAgentSystem,
    evaluator: TorchLinDistFlowEvaluator,
    test_loader: Any,
    adjacency: torch.Tensor,
    device: torch.device,
    rounds: int,
    initial_forecaster_path: Path,
    initial_surrogate_path: Path,
    strategy_checkpoint_dir: Path,
) -> np.ndarray:
    costs: list[float] = []
    for stage in range(1, rounds + 1):
        load_module_checkpoint(system.forecaster, initial_forecaster_path)
        load_module_checkpoint(system.surrogate, initial_surrogate_path)
        if strategy == "forecast-only":
            load_module_checkpoint(
                system.forecaster,
                strategy_checkpoint_dir / f"forecaster_only_{stage:02d}.pt",
            )
        elif strategy == "surrogate-only":
            load_module_checkpoint(
                system.surrogate,
                strategy_checkpoint_dir / f"surrogate_only_{stage:02d}.pt",
            )
        else:
            raise ValueError(f"Unsupported saved strategy: {strategy}")
        costs.append(
            evaluate_operating_cost(
                system.forecaster,
                system.surrogate,
                evaluator,
                test_loader,
                adjacency,
                device,
            )
        )
    return np.asarray(costs, dtype=np.float64)


def plot_strategy_comparison(
    costs: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["hatch.linewidth"] = 0.45
    plt.rcParams["svg.fonttype"] = "path"

    rounds = len(next(iter(costs.values())))
    x = np.arange(rounds)
    width = 0.21
    offset = 0.25
    styles = [
        ("Forecast-only", "#7199AF", "#3C6175", ""),
        ("Surrogate-only", "#CCD5DC", "#657783", ""),
        ("Alternating", "#78B3A3", "#376F60", ""),
    ]

    # 89 mm width: one column in a typical two-column manuscript.
    fig, (ax, base) = plt.subplots(
        2, 1, sharex=True, figsize=(3.5, 2.65), dpi=300,
        gridspec_kw={"height_ratios": [6, 1], "hspace": 0.09},
    )
    for index, (label, color, hatch_color, hatch) in enumerate(styles):
        values = costs[label]
        for axis in (ax, base):
            axis.bar(
                x + (index - 1) * offset, values, width,
                label=label, facecolor=color, edgecolor=hatch_color,
                linewidth=0.45, zorder=3,
            )

    all_values = np.concatenate(list(costs.values()))
    lower = np.floor((float(all_values.min()) - 20) / 50) * 50
    upper = np.ceil((float(all_values.max()) + 20) / 50) * 50
    ax.set_ylim(lower, upper)
    ax.set_yticks(np.arange(lower, upper + 1, 100))
    base.set_ylim(0, 100)
    base.set_yticks([0])
    base.set_xticks(x, [str(index) for index in range(1, rounds + 1)])
    ax.set_xlim(-0.58, rounds - 0.42)
    base.set_xlabel("Training round", fontsize=8.5, labelpad=4)
    fig.text(0.035, 0.51, "Dispatch cost", fontsize=8.5,
             rotation=90, va="center", ha="center")
    ax.grid(axis="y", color="#E9E9E9", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", labelsize=8, width=0.6, length=2.5, direction="out")
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    base.spines["top"].set_visible(False)
    base.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        base.spines[side].set_color("#333333")
        base.spines[side].set_linewidth(0.65)
    base.tick_params(labelsize=8, width=0.6, length=2.5)
    # Explicit break marks disclose the omitted interval on the cost axis.
    for axis, y in ((ax, 0), (base, 1)):
        axis.plot([0], [y], transform=axis.transAxes, clip_on=False,
                  marker=[(-1, -0.5), (1, 0.5)], markersize=7,
                  linestyle="none", color="#333333", mew=0.8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=False,
        fontsize=7.2,
        handlelength=1.25,
        handletextpad=0.35,
        columnspacing=0.7,
    )

    fig.subplots_adjust(left=0.18, right=0.985, top=0.84, bottom=0.18)
    fig.savefig(output_path, format="svg", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ieee13.yaml")
    parser.add_argument("--data", default="data/ieee13/ieee13.npz")
    parser.add_argument("--phase3-checkpoint-dir", default="checkpoints/phase3_states")
    parser.add_argument(
        "--output-dir",
        default="figures/phase3_training_strategy_comparison",
    )
    parser.add_argument(
        "--strategy-checkpoint-dir",
        default="checkpoints/phase3_training_strategy_comparison",
        help="Validation-best checkpoints for the two single-module baselines.",
    )
    parser.add_argument(
        "--strategy",
        choices=("all", "forecast-only", "surrogate-only", "plot"),
        default="all",
        help="Train one strategy, train both, or plot saved checkpoints.",
    )
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_dir = Path(args.phase3_checkpoint_dir)
    strategy_checkpoint_dir = Path(args.strategy_checkpoint_dir)
    strategy_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rounds = discover_phase3_rounds(checkpoint_dir)
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
    evaluator = make_evaluator(config, device)

    forecaster_paths = [
        checkpoint_dir / f"forecaster_{stage:02d}.pt"
        for stage in range(rounds + 1)
    ]
    surrogate_paths = [
        checkpoint_dir / f"surrogate_{stage:02d}.pt"
        for stage in range(rounds + 1)
    ]
    initial_system = build_system(config, nodes, device)
    load_module_checkpoint(initial_system.forecaster, forecaster_paths[0])
    load_module_checkpoint(initial_system.surrogate, surrogate_paths[0])
    baseline_cost = evaluate_operating_cost(
        initial_system.forecaster,
        initial_system.surrogate,
        evaluator,
        test_loader,
        adjacency,
        device,
    )

    if args.strategy == "surrogate-only":
        forecast_only_costs = None
    elif args.strategy in {"all", "forecast-only"}:
        forecast_only_system = build_system(config, nodes, device)
        forecast_only_system.forecaster.load_state_dict(
            initial_system.forecaster.state_dict()
        )
        forecast_only_system.surrogate.load_state_dict(
            initial_system.surrogate.state_dict()
        )
        forecast_only_costs = train_forecast_only(
            config,
            forecast_only_system,
            evaluator,
            train_loader,
            validation_loader,
            test_loader,
            adjacency,
            device,
            rounds,
            strategy_checkpoint_dir,
        )
    elif args.strategy == "plot":
        forecast_only_costs = evaluate_saved_single_strategy(
            "forecast-only",
            initial_system,
            evaluator,
            test_loader,
            adjacency,
            device,
            rounds,
            forecaster_paths[0],
            surrogate_paths[0],
            strategy_checkpoint_dir,
        )

    if args.strategy == "forecast-only":
        print("Forecast-only=" + " ".join(f"{value:.4f}" for value in forecast_only_costs))
        return

    if args.strategy in {"all", "surrogate-only"}:
        surrogate_only_system = build_system(
            config, nodes, device, disable_dropout=False
        )
        surrogate_only_system.forecaster.load_state_dict(
            initial_system.forecaster.state_dict()
        )
        surrogate_only_system.surrogate.load_state_dict(
            initial_system.surrogate.state_dict()
        )
        surrogate_only_costs = train_surrogate_only(
            config,
            surrogate_only_system,
            evaluator,
            train_loader,
            validation_loader,
            test_loader,
            adjacency,
            device,
            rounds,
            strategy_checkpoint_dir,
        )
    else:
        surrogate_only_costs = evaluate_saved_single_strategy(
            "surrogate-only",
            initial_system,
            evaluator,
            test_loader,
            adjacency,
            device,
            rounds,
            forecaster_paths[0],
            surrogate_paths[0],
            strategy_checkpoint_dir,
        )

    if args.strategy == "surrogate-only":
        print("Surrogate-only=" + " ".join(f"{value:.4f}" for value in surrogate_only_costs))
        return

    alternating_system = build_system(config, nodes, device)
    alternating_costs: list[float] = []
    for stage in range(1, rounds + 1):
        load_module_checkpoint(
            alternating_system.forecaster, forecaster_paths[stage]
        )
        load_module_checkpoint(
            alternating_system.surrogate, surrogate_paths[stage]
        )
        alternating_costs.append(
            evaluate_operating_cost(
                alternating_system.forecaster,
                alternating_system.surrogate,
                evaluator,
                test_loader,
                adjacency,
                device,
            )
        )

    raw_costs = {
        "Forecast-only": forecast_only_costs,
        "Surrogate-only": surrogate_only_costs,
        "Alternating": np.asarray(alternating_costs, dtype=np.float64),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase3_training_strategy_comparison.svg"
    plot_strategy_comparison(raw_costs, output_path)

    print(f"device={device}")
    print(f"rounds={rounds} baseline_operating_cost={baseline_cost:.6f}")
    for label, values in raw_costs.items():
        print(label + "=" + " ".join(f"{value:.4f}" for value in values))
    print(f"output={output_path.resolve()}")


if __name__ == "__main__":
    main()
