from __future__ import annotations

from pathlib import Path

from dual_agent.config import load_config


def test_load_config_resolves_dss_path_from_cwd() -> None:
    config = load_config("configs/ieee13.yaml")
    assert config.problem.samples == 1024
    assert config.problem.history_steps == 48
    assert config.problem.horizon_steps == 24
    assert config.problem.scenarios == 20
    assert config.problem.generate_dataset_scenarios is False
    assert config.network.dss_master == (Path.cwd() / "data/ieee13/IEEE13Nodeckt.dss").resolve()
    assert config.lindistflow.voltage_lower == 0.95
    assert config.lindistflow.energy_price_profile_per_kwh is not None
    assert len(config.lindistflow.energy_price_profile_per_kwh) == 24
    assert config.lindistflow.curtailment_per_kwh == 1
    assert config.lindistflow.terminal_soc_tolerance == 0.05
    assert len(config.network.batteries) == 2
    assert config.network.batteries[0].name == "bat_675"
    assert config.network.batteries[1].name == "bat_680"
    assert config.network.battery == config.network.batteries[0]
    assert config.training.forecaster_epochs == 500
    assert config.training.surrogate_epochs == 1000
    assert config.training.phase3_mode == "alternating"
    assert config.training.phase3_forecaster_epochs == 500
    assert config.training.phase3_surrogate_epochs == 100
    assert config.training.phase3_rounds == 5
    assert config.training.joint_epochs == 500
    assert config.training.joint_training_mode == "alternating"
    assert config.training.joint_alternating_updates is True
    assert config.training.joint_rounds == 5
    assert config.training.joint_surrogate_epochs_per_round == 20
    assert config.training.batch_size == 512
    assert config.training.learning_rate == 0.0002
    assert config.training.joint_learning_rate == 0.0002
    assert config.training.joint_surrogate_learning_rate == 0.0002
    assert config.training.warmup_epochs == 20
    assert config.training.min_learning_rate_ratio == 0.1
    assert config.training.max_grad_norm == 1.0
    assert config.training.joint_forecast_loss_tolerance == 0.2
    assert config.training.joint_forecast_constraint_weight == 0.2
    assert config.training.operating_cost_loss_weight == 0.5
    assert config.training.voltage_violation_loss_weight == 10
    assert config.training.line_flow_violation_loss_weight == 10
    assert config.training.kw_violation_loss_weight == 0
    assert config.training.soc_violation_loss_weight == 0
    assert config.training.terminal_soc_loss_weight == 200
