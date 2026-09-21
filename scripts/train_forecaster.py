from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from dual_agent.config import load_config
from dual_agent.models.forecaster import PVScenarioForecaster
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
    train_forecaster_epoch,
)
from dual_agent.training.tracking import init_swanlab_tracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ieee13.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="forecaster.pt")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--swanlab", action="store_true", help="Track training metrics with SwanLab.")
    parser.add_argument("--swanlab-project", default="dual-agent-opf")
    parser.add_argument("--swanlab-experiment", default="forecaster")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = ScenarioDataset(args.data)
    validate_dataset_matches_config(dataset, config)
    train_loader, test_loader = make_train_test_loaders(
        dataset,
        batch_size=config.training.batch_size,
        test_fraction=args.test_fraction,
        seed=config.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes = dataset.pv_history.size(2)
    adjacency = make_complete_graph(nodes)

    model = PVScenarioForecaster(
        pv_feature_dim=config.problem.pv_feature_dim,
        hidden_dim=config.model.hidden_dim,
        horizon_steps=config.problem.horizon_steps,
        scenarios=config.problem.scenarios,
        graph_layers=config.model.graph_layers,
        temporal_layers=config.model.temporal_layers,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        config.training.forecaster_epochs,
        len(train_loader),
        config.training.warmup_epochs,
        config.training.min_learning_rate_ratio,
    )
    tracker = init_swanlab_tracker(
        enabled=args.swanlab,
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment,
        config={
            "trainer": "forecaster",
            "config": config,
            "data": args.data,
            "checkpoint": args.checkpoint,
            "test_fraction": args.test_fraction,
            "device": str(device),
        },
    )

    try:
        for epoch in range(config.training.forecaster_epochs):
            train_loss = train_forecaster_epoch(
                model,
                train_loader,
                optimizer,
                adjacency,
                device,
                scheduler,
                config.training.max_grad_norm,
            )
            test_loss = evaluate_forecaster_loss(model, test_loader, adjacency, device)
            metrics = {
                "train/loss": train_loss,
                "test/loss": test_loss,
                "learning_rate": current_learning_rate(optimizer),
            }
            tracker.log(metrics, step=epoch + 1)
            print(f"epoch={epoch + 1:03d} train_loss={train_loss:.6f} test_loss={test_loss:.6f}")
    finally:
        tracker.finish()

    torch.save(model.state_dict(), args.checkpoint)


if __name__ == "__main__":
    main()
