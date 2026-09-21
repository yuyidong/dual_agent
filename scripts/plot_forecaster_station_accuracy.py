from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ieee13.yaml")
    parser.add_argument("--data", default="data/ieee13.npz")
    parser.add_argument("--before-checkpoint", default="forecaster.pt")
    parser.add_argument("--surrogate-checkpoint", default="surrogate.pt")
    parser.add_argument("--after-checkpoint", default="dual_agent.pt")
    parser.add_argument("--out", default="data/pv_forecast_station_accuracy_before_after.png")
    parser.add_argument("--test-fraction", type=float, default=0.2)
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

    before = _load_forecaster(args.before_checkpoint, config, nodes, device, prefix="")
    after = _load_forecaster(args.after_checkpoint, config, nodes, device, prefix="forecaster.")
    before_metrics = _station_forecast_metrics(before, test_loader, adjacency, device)
    after_metrics = _station_forecast_metrics(after, test_loader, adjacency, device)
    before_surrogate = _make_surrogate(config, nodes, device)
    before_surrogate.load_state_dict(torch.load(args.surrogate_checkpoint, map_location=device))
    before_surrogate.eval()
    after_system = DualAgentSystem(_make_forecaster(config, device), _make_surrogate(config, nodes, device)).to(device)
    after_system.load_state_dict(torch.load(args.after_checkpoint, map_location=device))
    after_system.eval()
    evaluator = TorchLinDistFlowEvaluator(
        config.network.dss_master,
        config.network.pv_systems,
        config.network.batteries,
        config.lindistflow,
        config.problem.interval_hours,
    ).to(device)
    evaluator.eval()
    before_decision, after_decision = _decision_metrics(
        before,
        before_surrogate,
        after_system,
        evaluator,
        test_loader,
        adjacency,
        device,
    )

    station_labels = list(config.network.pv_systems)
    _plot_station_errors(
        station_labels,
        before_metrics,
        after_metrics,
        before_decision,
        after_decision,
        Path(args.out),
    )
    print(f"saved_plot={Path(args.out).resolve()}")
    for idx, label in enumerate(station_labels):
        print(
            f"{label}: "
            f"actual_mean={before_metrics['target_mean'][idx]:.3f} "
            f"before_mean={before_metrics['forecast_mean'][idx]:.3f} "
            f"after_mean={after_metrics['forecast_mean'][idx]:.3f} "
            f"before_crps={before_metrics['crps'][idx]:.3f} after_crps={after_metrics['crps'][idx]:.3f}"
        )
    print(
        "decision_metrics: "
        f"actual_total_mean={before_metrics['target_mean'].sum():.3f} "
        f"before_total_mean={before_metrics['forecast_mean'].sum():.3f} "
        f"after_total_mean={after_metrics['forecast_mean'].sum():.3f} "
        f"before_operating_cost={before_decision['operating_cost']:.3f} "
        f"after_operating_cost={after_decision['operating_cost']:.3f} "
        f"before_curtailment_cost={before_decision['curtailment_cost']:.3f} "
        f"after_curtailment_cost={after_decision['curtailment_cost']:.3f}"
    )


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


def _load_forecaster(
    checkpoint: str,
    config,
    nodes: int,
    device: torch.device,
    prefix: str,
) -> PVScenarioForecaster:
    model = _make_forecaster(config, device)
    state = torch.load(checkpoint, map_location=device)
    if prefix:
        state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _station_forecast_metrics(
    model: PVScenarioForecaster,
    test_loader,
    adjacency: torch.Tensor,
    device: torch.device,
) -> dict[str, np.ndarray]:
    crps_sum: torch.Tensor | None = None
    forecast_sum: torch.Tensor | None = None
    target_sum: torch.Tensor | None = None
    count = 0
    for batch in test_loader:
        history = batch["pv_history"].to(device)
        target = batch["pv_target"].to(device)
        prediction = model(history, adjacency)
        scenario_mean = prediction.mean(dim=1)
        batch_crps = _station_crps(prediction, target).sum(dim=(0, 1))
        batch_forecast = scenario_mean.sum(dim=(0, 1))
        batch_target = target.sum(dim=(0, 1))
        crps_sum = batch_crps if crps_sum is None else crps_sum + batch_crps
        forecast_sum = batch_forecast if forecast_sum is None else forecast_sum + batch_forecast
        target_sum = batch_target if target_sum is None else target_sum + batch_target
        count += target.size(0) * target.size(1)

    if crps_sum is None or forecast_sum is None or target_sum is None:
        raise ValueError("test loader is empty")
    crps = crps_sum / count
    return {
        "crps": crps.cpu().numpy(),
        "forecast_mean": (forecast_sum / count).cpu().numpy(),
        "target_mean": (target_sum / count).cpu().numpy(),
    }


def _station_crps(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target[:, None, :, :]
    term_1 = (prediction - target).abs().mean(dim=1)
    pairwise = prediction[:, :, None, :, :] - prediction[:, None, :, :, :]
    term_2 = pairwise.abs().mean(dim=(1, 2))
    return term_1 - 0.5 * term_2


@torch.no_grad()
def _decision_metrics(
    before_forecaster: PVScenarioForecaster,
    before_surrogate: OPFSurrogate,
    after_system: DualAgentSystem,
    evaluator: TorchLinDistFlowEvaluator,
    test_loader,
    adjacency: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    before_totals = {"operating_cost": 0.0, "curtailment_cost": 0.0}
    after_totals = {"operating_cost": 0.0, "curtailment_cost": 0.0}
    samples = 0
    for batch in test_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        before_scenarios = before_forecaster(batch["pv_history"], adjacency)
        before_output = before_surrogate(before_scenarios, batch["load_forecast"], adjacency)
        after_output = after_system(batch["pv_history"], batch["load_forecast"], adjacency)
        before_result = evaluator(
            before_output["dispatch"],
            batch["pv_target"].unsqueeze(1),
            batch["load_forecast"],
        )
        after_result = evaluator(
            after_output["dispatch"],
            batch["pv_target"].unsqueeze(1),
            batch["load_forecast"],
        )
        batch_size = batch["pv_history"].size(0)
        samples += batch_size
        for key in before_totals:
            before_totals[key] += float(before_result[key].detach().sum().cpu())
            after_totals[key] += float(after_result[key].detach().sum().cpu())

    if samples == 0:
        raise ValueError("test loader is empty")
    return (
        {key: value / samples for key, value in before_totals.items()},
        {key: value / samples for key, value in after_totals.items()},
    )


def _plot_station_errors(
    labels: list[str],
    before_metrics: dict[str, np.ndarray],
    after_metrics: dict[str, np.ndarray],
    before_decision: dict[str, float],
    after_decision: dict[str, float],
    out: Path,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    width = 0.26

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].bar(x - width / 2, before_metrics["crps"], width, label="Before phase 3", color="#6aaed6")
    axes[0].bar(x + width / 2, after_metrics["crps"], width, label="After phase 3", color="#154c9f")
    axes[0].set_title("CRPS by PV station")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("Forecast error (kW)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    total_labels = ["Actual", "Before phase 3", "After phase 3"]
    total_values = [
        before_metrics["target_mean"].sum(),
        before_metrics["forecast_mean"].sum(),
        after_metrics["forecast_mean"].sum(),
    ]
    axes[1].bar(total_labels, total_values, color=["#444444", "#6aaed6", "#154c9f"])
    axes[1].set_title("Mean total PV forecast")
    axes[1].set_ylabel("Average total power (kW)")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.25)

    decision_labels = ["Operating cost", "Curtailment penalty"]
    before_values = [before_decision["operating_cost"], before_decision["curtailment_cost"]]
    after_values = [after_decision["operating_cost"], after_decision["curtailment_cost"]]
    decision_x = np.arange(len(decision_labels))
    axes[2].bar(decision_x - width / 2, before_values, width, label="Before phase 3", color="#6aaed6")
    axes[2].bar(decision_x + width / 2, after_values, width, label="After phase 3", color="#154c9f")
    axes[2].set_title("Decision metrics on test set")
    axes[2].set_xticks(decision_x)
    axes[2].set_xticklabels(decision_labels, rotation=20, ha="right")
    axes[2].set_ylabel("Average cost")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()

    fig.suptitle("PV Forecast Accuracy Before vs After Phase 3 Joint Training")
    fig.tight_layout()
    fig.savefig(out, dpi=180)


if __name__ == "__main__":
    main()
