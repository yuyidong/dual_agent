from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BatteryConfig:
    name: str
    bus: str
    phases: int
    kv: float
    kw_rated: float
    kwh_rated: float
    soc_initial: float
    soc_min: float
    soc_max: float
    charge_efficiency: float
    discharge_efficiency: float


@dataclass(frozen=True)
class NetworkConfig:
    dss_master: Path
    pv_systems: tuple[str, ...]
    load_buses: tuple[str, ...]
    batteries: tuple[BatteryConfig, ...]

    @property
    def battery(self) -> BatteryConfig:
        return self.batteries[0]


@dataclass(frozen=True)
class ProblemConfig:
    samples: int
    history_steps: int
    horizon_steps: int
    scenarios: int
    generate_dataset_scenarios: bool
    interval_hours: float
    pv_feature_dim: int
    load_feature_dim: int


@dataclass(frozen=True)
class LinDistFlowConfig:
    voltage_lower: float
    voltage_upper: float
    degradation_per_kwh: float
    curtailment_per_kwh: float
    energy_price_per_kwh: float = 0.12
    energy_price_profile_per_kwh: tuple[float, ...] | None = None
    terminal_soc_tolerance: float = 0.05


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    graph_layers: int
    temporal_layers: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    forecaster_epochs: int
    surrogate_epochs: int
    phase3_mode: str
    phase3_forecaster_epochs: int
    phase3_surrogate_epochs: int
    phase3_rounds: int
    joint_epochs: int
    joint_training_mode: str
    joint_alternating_updates: bool
    joint_rounds: int
    joint_surrogate_epochs_per_round: int
    learning_rate: float
    joint_learning_rate: float
    joint_surrogate_learning_rate: float
    warmup_epochs: int
    min_learning_rate_ratio: float
    max_grad_norm: float | None
    joint_forecast_loss_tolerance: float
    joint_forecast_constraint_weight: float
    joint_forecast_loss_weight: float
    phase3_reset_forecaster_scheduler_per_round: bool
    operating_cost_loss_weight: float
    voltage_violation_loss_weight: float
    line_flow_violation_loss_weight: float
    kw_violation_loss_weight: float
    soc_violation_loss_weight: float
    terminal_soc_loss_weight: float
    surrogate_consistency_loss_weight: float
    phase3_early_stopping_patience: int
    phase3_early_stopping_min_delta: float
    phase3_validation_interval: int
    phase3_disable_dropout: bool
    phase3_forecaster_first: bool
    phase3_validation_fraction: float


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    network: NetworkConfig
    problem: ProblemConfig
    lindistflow: LinDistFlowConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    cwd = Path.cwd()

    battery_entries = raw["network"].get("batteries")
    if battery_entries is None:
        battery_entries = [raw["network"]["battery"]]
    batteries = tuple(BatteryConfig(**entry) for entry in battery_entries)
    if not batteries:
        raise ValueError("network must define at least one battery.")
    network = NetworkConfig(
        dss_master=(cwd / raw["network"]["dss_master"]).resolve()
        if not Path(raw["network"]["dss_master"]).is_absolute()
        else Path(raw["network"]["dss_master"]),
        pv_systems=tuple(raw["network"]["pv_systems"]),
        load_buses=tuple(str(bus) for bus in raw["network"]["load_buses"]),
        batteries=batteries,
    )

    training_raw = dict(raw["training"])
    training_raw.setdefault("joint_learning_rate", training_raw["learning_rate"])
    training_raw.setdefault("joint_surrogate_learning_rate", training_raw["learning_rate"])
    training_raw.setdefault("warmup_epochs", 0)
    training_raw.setdefault("min_learning_rate_ratio", 0.1)
    training_raw.setdefault("max_grad_norm", None)
    training_raw.setdefault(
        "joint_alternating_updates",
        training_raw.get("phase3_mode") == "alternating",
    )
    training_raw.setdefault("joint_rounds", training_raw.get("phase3_rounds", 1))
    training_raw.setdefault(
        "joint_epochs",
        training_raw.get("phase3_forecaster_epochs", training_raw.get("phase3_surrogate_epochs", 0)),
    )
    if "joint_surrogate_epochs_per_round" not in training_raw:
        phase3_surrogate_epochs = int(training_raw.get("phase3_surrogate_epochs", 1))
        phase3_rounds = int(training_raw.get("phase3_rounds", training_raw["joint_rounds"]))
        training_raw["joint_surrogate_epochs_per_round"] = max(
            1,
            phase3_surrogate_epochs // max(1, phase3_rounds),
        )
    training_raw.setdefault(
        "joint_training_mode",
        training_raw.get(
            "phase3_mode",
            "alternating" if bool(training_raw["joint_alternating_updates"]) else "forecast_only",
        ),
    )
    training_raw.setdefault("phase3_mode", training_raw["joint_training_mode"])
    training_raw.setdefault("phase3_forecaster_epochs", training_raw["joint_epochs"])
    training_raw.setdefault(
        "phase3_surrogate_epochs",
        int(training_raw["joint_rounds"]) * int(training_raw["joint_surrogate_epochs_per_round"])
        if training_raw["phase3_mode"] == "alternating"
        else training_raw["joint_epochs"],
    )
    training_raw.setdefault("phase3_rounds", training_raw["joint_rounds"])
    training_raw.setdefault("joint_forecast_loss_tolerance", 0.05)
    training_raw.setdefault("joint_forecast_constraint_weight", 1.0)
    training_raw.setdefault("joint_forecast_loss_weight", 0.0)
    training_raw.setdefault("phase3_reset_forecaster_scheduler_per_round", False)
    training_raw.setdefault("surrogate_consistency_loss_weight", 0.0)
    training_raw.setdefault("phase3_early_stopping_patience", 0)
    training_raw.setdefault("phase3_early_stopping_min_delta", 0.001)
    training_raw.setdefault("phase3_validation_interval", 10)
    training_raw.setdefault("phase3_disable_dropout", False)
    training_raw.setdefault("phase3_forecaster_first", False)
    training_raw.setdefault("phase3_validation_fraction", 0.2)
    training_raw.setdefault(
        "operating_cost_loss_weight",
        training_raw.get("decision_loss_weight", 1.0),
    )
    legacy_network_weight = training_raw.pop("network_violation_loss_weight", None)
    training_raw.setdefault(
        "voltage_violation_loss_weight",
        legacy_network_weight if legacy_network_weight is not None else training_raw.get("risk_loss_weight", 1.0),
    )
    training_raw.setdefault(
        "line_flow_violation_loss_weight",
        legacy_network_weight if legacy_network_weight is not None else training_raw.get("risk_loss_weight", 1.0),
    )
    training_raw.setdefault(
        "kw_violation_loss_weight",
        training_raw.get("risk_loss_weight", training_raw["voltage_violation_loss_weight"]),
    )
    training_raw.setdefault(
        "soc_violation_loss_weight",
        training_raw.get("soc_loss_weight", training_raw["voltage_violation_loss_weight"]),
    )
    training_raw.setdefault(
        "terminal_soc_loss_weight",
        training_raw.get("terminal_soc_loss_weight", training_raw["soc_violation_loss_weight"]),
    )
    training_raw.pop("decision_loss_weight", None)
    training_raw.pop("risk_loss_weight", None)
    training_raw.pop("soc_loss_weight", None)
    if float(training_raw["joint_forecast_loss_weight"]) < 0:
        raise ValueError("training.joint_forecast_loss_weight must be non-negative.")
    if float(training_raw["surrogate_consistency_loss_weight"]) < 0:
        raise ValueError("training.surrogate_consistency_loss_weight must be non-negative.")
    if int(training_raw["phase3_early_stopping_patience"]) < 0:
        raise ValueError("training.phase3_early_stopping_patience must be non-negative.")
    if float(training_raw["phase3_early_stopping_min_delta"]) < 0:
        raise ValueError("training.phase3_early_stopping_min_delta must be non-negative.")
    if int(training_raw["phase3_validation_interval"]) < 1:
        raise ValueError("training.phase3_validation_interval must be at least 1.")
    if not 0.0 < float(training_raw["phase3_validation_fraction"]) < 1.0:
        raise ValueError("training.phase3_validation_fraction must be between 0 and 1.")
    if int(training_raw["joint_rounds"]) < 1:
        raise ValueError("training.joint_rounds must be at least 1.")
    if int(training_raw["joint_surrogate_epochs_per_round"]) < 1:
        raise ValueError("training.joint_surrogate_epochs_per_round must be at least 1.")
    if int(training_raw["warmup_epochs"]) < 0:
        raise ValueError("training.warmup_epochs must be non-negative.")
    if float(training_raw["min_learning_rate_ratio"]) < 0:
        raise ValueError("training.min_learning_rate_ratio must be non-negative.")
    if training_raw["max_grad_norm"] is not None and float(training_raw["max_grad_norm"]) <= 0:
        raise ValueError("training.max_grad_norm must be positive when set.")
    valid_joint_modes = {"forecast_only", "surrogate_only", "alternating"}
    if training_raw["joint_training_mode"] not in valid_joint_modes:
        raise ValueError(
            "training.joint_training_mode must be one of: "
            + ", ".join(sorted(valid_joint_modes))
            + "."
        )
    if training_raw["phase3_mode"] not in valid_joint_modes:
        raise ValueError(
            "training.phase3_mode must be one of: "
            + ", ".join(sorted(valid_joint_modes))
            + "."
        )
    if int(training_raw["phase3_forecaster_epochs"]) < 0:
        raise ValueError("training.phase3_forecaster_epochs must be non-negative.")
    if int(training_raw["phase3_surrogate_epochs"]) < 0:
        raise ValueError("training.phase3_surrogate_epochs must be non-negative.")
    if int(training_raw["phase3_rounds"]) < 1:
        raise ValueError("training.phase3_rounds must be at least 1.")
    if (
        training_raw["phase3_mode"] == "alternating"
        and int(training_raw["phase3_forecaster_epochs"]) < int(training_raw["phase3_rounds"])
    ):
        raise ValueError("training.phase3_forecaster_epochs must be at least training.phase3_rounds.")

    problem_raw = dict(raw["problem"])
    problem_raw.setdefault("samples", 365)
    problem_raw.setdefault("generate_dataset_scenarios", False)

    lindistflow_raw = dict(raw.get("lindistflow", raw.get("teacher")))
    lindistflow_raw.pop("voltage_violation_weight", None)
    profile = lindistflow_raw.get("energy_price_profile_per_kwh")
    lindistflow_raw["energy_price_profile_per_kwh"] = tuple(profile) if profile is not None else None

    return ExperimentConfig(
        seed=int(raw["seed"]),
        network=network,
        problem=ProblemConfig(**problem_raw),
        lindistflow=LinDistFlowConfig(**lindistflow_raw),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**training_raw),
    )


def asdict_shallow(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "network": config.network,
        "problem": config.problem,
        "lindistflow": config.lindistflow,
        "model": config.model,
        "training": config.training,
    }
