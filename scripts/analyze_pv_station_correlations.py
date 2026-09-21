from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

import matplotlib.pyplot as plt
import numpy as np
import torch

from dual_agent.config import load_config
from dual_agent.models.forecaster import PVScenarioForecaster
from dual_agent.training.dataset import ScenarioDataset, make_complete_graph, make_train_test_loaders
from dual_agent.training.dataset import validate_dataset_matches_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ieee13.yaml")
    parser.add_argument("--data", default="data/ieee13.npz")
    parser.add_argument("--before-checkpoint", default="forecaster.pt")
    parser.add_argument("--after-checkpoint", default="dual_agent.pt")
    parser.add_argument("--out", default="data/pv_station_correlation_analysis.png")
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

    before = _load_forecaster(args.before_checkpoint, config, device, prefix="")
    after = _load_forecaster(args.after_checkpoint, config, device, prefix="forecaster.")
    actual, before_error, after_error = _collect_station_series(before, after, test_loader, adjacency, device)

    matrices = {
        "Actual PV": _corrcoef(actual),
        "Before phase 3 error": _corrcoef(before_error),
        "After phase 3 error": _corrcoef(after_error),
    }
    labels = list(config.network.pv_systems)
    _plot_matrices(matrices, labels, Path(args.out))
    print(f"saved_plot={Path(args.out).resolve()}")
    for name, matrix in matrices.items():
        print(name)
        print(np.array2string(matrix, precision=3, suppress_small=True))


def _load_forecaster(checkpoint: str, config, device: torch.device, prefix: str) -> PVScenarioForecaster:
    model = PVScenarioForecaster(
        pv_feature_dim=config.problem.pv_feature_dim,
        hidden_dim=config.model.hidden_dim,
        horizon_steps=config.problem.horizon_steps,
        scenarios=config.problem.scenarios,
        graph_layers=config.model.graph_layers,
        temporal_layers=config.model.temporal_layers,
        dropout=config.model.dropout,
    ).to(device)
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
def _collect_station_series(
    before: PVScenarioForecaster,
    after: PVScenarioForecaster,
    test_loader,
    adjacency: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual_parts = []
    before_error_parts = []
    after_error_parts = []
    for batch in test_loader:
        history = batch["pv_history"].to(device)
        target = batch["pv_target"].to(device)
        before_mean = before(history, adjacency).mean(dim=1)
        after_mean = after(history, adjacency).mean(dim=1)
        actual_parts.append(_flatten_time(target))
        before_error_parts.append(_flatten_time(before_mean - target))
        after_error_parts.append(_flatten_time(after_mean - target))

    return (
        np.concatenate(actual_parts, axis=0),
        np.concatenate(before_error_parts, axis=0),
        np.concatenate(after_error_parts, axis=0),
    )


def _flatten_time(series: torch.Tensor) -> np.ndarray:
    return series.detach().cpu().reshape(-1, series.size(-1)).numpy()


def _corrcoef(series: np.ndarray) -> np.ndarray:
    if series.ndim != 2:
        raise ValueError("series must have shape [samples_times_horizon, stations]")
    return np.corrcoef(series, rowvar=False)


def _plot_matrices(matrices: dict[str, np.ndarray], labels: list[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(matrices), figsize=(14, 4.2), constrained_layout=True)
    for ax, (title, matrix) in zip(axes, matrices.items(), strict=True):
        image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticklabels(labels)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes, shrink=0.82, label="Correlation")
    fig.suptitle("PV Station Correlation Analysis on Test Set")
    fig.savefig(out, dpi=180)


if __name__ == "__main__":
    main()
