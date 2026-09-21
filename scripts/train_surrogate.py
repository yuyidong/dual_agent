from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from dual_agent.config import load_config
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
    evaluate_surrogate_loss,
    make_warmup_cosine_scheduler,
    train_surrogate_epoch,
)
from dual_agent.training.tracking import init_swanlab_tracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ieee13.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--forecaster-checkpoint", default=None)
    parser.add_argument("--checkpoint", default="surrogate.pt")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--swanlab", action="store_true", help="Track training metrics with SwanLab.")
    parser.add_argument("--swanlab-project", default="dual-agent-opf")
    parser.add_argument("--swanlab-experiment", default="surrogate")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = ScenarioDataset(args.data)
    validate_dataset_matches_config(
        dataset,
        config,
        require_scenarios=args.forecaster_checkpoint is None,
    )
    train_loader, test_loader = make_train_test_loaders(
        dataset,
        batch_size=config.training.batch_size,
        test_fraction=args.test_fraction,
        seed=config.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes = dataset.pv_history.size(2)
    adjacency = make_complete_graph(nodes)

    forecaster = None
    if args.forecaster_checkpoint is not None:
        forecaster = PVScenarioForecaster(
            pv_feature_dim=config.problem.pv_feature_dim,
            hidden_dim=config.model.hidden_dim,
            horizon_steps=config.problem.horizon_steps,
            scenarios=config.problem.scenarios,
            graph_layers=config.model.graph_layers,
            temporal_layers=config.model.temporal_layers,
            dropout=config.model.dropout,
        ).to(device)
        forecaster.load_state_dict(torch.load(args.forecaster_checkpoint, map_location=device))
        forecaster.eval()
        for param in forecaster.parameters():
            param.requires_grad_(False)

    model = OPFSurrogate(
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
    evaluator = TorchLinDistFlowEvaluator(
        config.network.dss_master,
        config.network.pv_systems,
        config.network.batteries,
        config.lindistflow,
        config.problem.interval_hours,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        config.training.surrogate_epochs,
        len(train_loader),
        config.training.warmup_epochs,
        config.training.min_learning_rate_ratio,
    )
    tracker = init_swanlab_tracker(
        enabled=args.swanlab,
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment,
        config={
            "trainer": "surrogate",
            "config": config,
            "data": args.data,
            "forecaster_checkpoint": args.forecaster_checkpoint,
            "checkpoint": args.checkpoint,
            "test_fraction": args.test_fraction,
            "device": str(device),
        },
    )

    best_test_loss = float("inf")
    try:
        for epoch in range(config.training.surrogate_epochs):
            train_metrics = train_surrogate_epoch(
                model,
                evaluator,
                train_loader,
                optimizer,
                adjacency,
                device,
                forecaster,
                config.training.operating_cost_loss_weight,
                config.training.voltage_violation_loss_weight,
                config.training.line_flow_violation_loss_weight,
                config.training.kw_violation_loss_weight,
                config.training.soc_violation_loss_weight,
                config.training.terminal_soc_loss_weight,
                scheduler,
                config.training.max_grad_norm,
            )
            train_loss = train_metrics["loss"]
            test_metrics = evaluate_surrogate_loss(
                model,
                evaluator,
                test_loader,
                adjacency,
                device,
                forecaster,
                config.training.operating_cost_loss_weight,
                config.training.voltage_violation_loss_weight,
                config.training.line_flow_violation_loss_weight,
                config.training.kw_violation_loss_weight,
                config.training.soc_violation_loss_weight,
                config.training.terminal_soc_loss_weight,
            )
            test_loss = test_metrics["loss"]
            metrics = {
                "train/loss": train_loss,
                "test/loss": test_loss,
                "learning_rate": current_learning_rate(optimizer),
                "train/lindistflow_operating_cost": train_metrics["operating_cost"],
                "train/lindistflow_curtailment_cost": train_metrics["curtailment_cost"],
                "train/battery_dispatch_abs_mean": train_metrics["dispatch_abs_mean"],
                "train/battery_dispatch_abs_max": train_metrics["dispatch_abs_max"],
                "train/battery_charge_energy_kwh": train_metrics["charge_energy_kwh"],
                "train/battery_discharge_energy_kwh": train_metrics["discharge_energy_kwh"],
                "train/battery_soc_min": train_metrics["soc_min"],
                "train/battery_soc_max": train_metrics["soc_max"],
                "train/lindistflow_voltage_violation": train_metrics["voltage_violation"],
                "train/lindistflow_line_flow_violation": train_metrics["line_flow_violation"],
                "train/lindistflow_network_violation": train_metrics["network_violation"],
                "train/lindistflow_kw_violation": train_metrics["kw_violation"],
                "train/lindistflow_soc_violation": train_metrics["soc_violation"],
                "train/lindistflow_terminal_soc_deviation": train_metrics["terminal_soc_deviation"],
                "test/lindistflow_operating_cost": test_metrics["operating_cost"],
                "test/lindistflow_curtailment_cost": test_metrics["curtailment_cost"],
                "test/battery_dispatch_abs_mean": test_metrics["dispatch_abs_mean"],
                "test/battery_dispatch_abs_max": test_metrics["dispatch_abs_max"],
                "test/battery_charge_energy_kwh": test_metrics["charge_energy_kwh"],
                "test/battery_discharge_energy_kwh": test_metrics["discharge_energy_kwh"],
                "test/battery_soc_min": test_metrics["soc_min"],
                "test/battery_soc_max": test_metrics["soc_max"],
                "test/lindistflow_voltage_violation": test_metrics["voltage_violation"],
                "test/lindistflow_line_flow_violation": test_metrics["line_flow_violation"],
                "test/lindistflow_network_violation": test_metrics["network_violation"],
                "test/lindistflow_kw_violation": test_metrics["kw_violation"],
                "test/lindistflow_soc_violation": test_metrics["soc_violation"],
                "test/lindistflow_terminal_soc_deviation": test_metrics["terminal_soc_deviation"],
            }
            tracker.log(metrics, step=epoch + 1)
            print(
                f"epoch={epoch + 1:03d} train_loss={train_loss:.6f} test_loss={test_loss:.6f} "
                f"operating_cost={test_metrics['operating_cost']:.6f} "
                f"curtailment_cost={test_metrics['curtailment_cost']:.6f} "
                f"dispatch_abs_mean={test_metrics['dispatch_abs_mean']:.6f} "
                f"soc_min={test_metrics['soc_min']:.6f} "
                f"soc_max={test_metrics['soc_max']:.6f} "
                f"voltage_violation={test_metrics['voltage_violation']:.6f} "
                f"line_flow_violation={test_metrics['line_flow_violation']:.6f} "
                f"kw_violation={test_metrics['kw_violation']:.6f} "
                f"soc_violation={test_metrics['soc_violation']:.6f} "
                f"terminal_soc_deviation={test_metrics['terminal_soc_deviation']:.6f}"
            )
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), args.checkpoint)
                print(f"saved_checkpoint={args.checkpoint} best_test_loss={best_test_loss:.6f}")
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
