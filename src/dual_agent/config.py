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
class DataSplitConfig:
    """Dataset partitioning policy shared by all training stages."""

    train_fraction: float
    validation_fraction: float
    test_fraction: float


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
    forecaster_learning_rate: float
    surrogate_learning_rate: float
    phase3_forecaster_learning_rate: float
    phase3_surrogate_learning_rate: float
    warmup_epochs: int
    min_learning_rate_ratio: float
    max_grad_norm: float | None
    joint_forecast_loss_tolerance: float
    joint_forecast_constraint_weight: float
    joint_forecast_loss_weight: float
    operating_cost_loss_weight: float
    voltage_violation_loss_weight: float
    line_flow_violation_loss_weight: float
    kw_violation_loss_weight: float
    soc_violation_loss_weight: float
    terminal_soc_loss_weight: float


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    network: NetworkConfig
    problem: ProblemConfig
    data_split: DataSplitConfig
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

    split_raw = dict(raw.get("data_split", {}))
    split_raw.setdefault("train_fraction", 0.64)
    split_raw.setdefault("validation_fraction", 0.16)
    split_raw.setdefault("test_fraction", 0.20)
    data_split = DataSplitConfig(**split_raw)
    fractions = (
        data_split.train_fraction,
        data_split.validation_fraction,
        data_split.test_fraction,
    )
    if any(fraction < 0.0 or fraction > 1.0 for fraction in fractions):
        raise ValueError("data_split fractions must be between 0 and 1.")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(
            "data_split.train_fraction + validation_fraction + "
            "test_fraction must equal 1."
        )
    if data_split.train_fraction <= 0.0 or data_split.test_fraction <= 0.0:
        raise ValueError("data_split train and test fractions must be positive.")

    training_raw = dict(raw["training"])
    legacy_learning_rate = training_raw.pop("learning_rate", None)
    training_raw.setdefault(
        "forecaster_learning_rate",
        legacy_learning_rate if legacy_learning_rate is not None else 0.0001,
    )
    training_raw.setdefault(
        "surrogate_learning_rate",
        legacy_learning_rate
        if legacy_learning_rate is not None
        else training_raw["forecaster_learning_rate"],
    )
    training_raw.setdefault(
        "phase3_forecaster_learning_rate",
        training_raw["forecaster_learning_rate"],
    )
    training_raw.setdefault(
        "phase3_surrogate_learning_rate",
        training_raw["surrogate_learning_rate"],
    )
    training_raw.setdefault("warmup_epochs", 0)
    training_raw.setdefault("min_learning_rate_ratio", 0.1)
    training_raw.setdefault("max_grad_norm", None)
    training_raw.setdefault("phase3_mode", "alternating")
    training_raw.setdefault(
        "phase3_forecaster_epochs",
        training_raw.get("forecaster_epochs", 0),
    )
    training_raw.setdefault(
        "phase3_surrogate_epochs",
        training_raw.get("surrogate_epochs", 0),
    )
    training_raw.setdefault("phase3_rounds", 1)
    training_raw.setdefault("joint_forecast_loss_tolerance", 0.05)
    training_raw.setdefault("joint_forecast_constraint_weight", 1.0)
    training_raw.setdefault("joint_forecast_loss_weight", 0.0)
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
    if int(training_raw["warmup_epochs"]) < 0:
        raise ValueError("training.warmup_epochs must be non-negative.")
    if float(training_raw["min_learning_rate_ratio"]) < 0:
        raise ValueError("training.min_learning_rate_ratio must be non-negative.")
    if training_raw["max_grad_norm"] is not None and float(training_raw["max_grad_norm"]) <= 0:
        raise ValueError("training.max_grad_norm must be positive when set.")
    valid_joint_modes = {"forecast_only", "surrogate_only", "alternating"}
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
        data_split=data_split,
        lindistflow=LinDistFlowConfig(**lindistflow_raw),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**training_raw),
    )


def asdict_shallow(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "network": config.network,
        "problem": config.problem,
        "data_split": config.data_split,
        "lindistflow": config.lindistflow,
        "model": config.model,
        "training": config.training,
    }
