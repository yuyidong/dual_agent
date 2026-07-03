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
    battery: BatteryConfig


@dataclass(frozen=True)
class ProblemConfig:
    history_steps: int
    horizon_steps: int
    scenarios: int
    interval_hours: float
    pv_feature_dim: int
    load_feature_dim: int


@dataclass(frozen=True)
class TeacherConfig:
    candidates: int
    elite_fraction: float
    cem_iterations: int
    voltage_lower: float
    voltage_upper: float
    energy_price_per_kwh: float
    degradation_per_kwh: float
    curtailment_per_kwh: float
    voltage_violation_weight: float


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    graph_layers: int
    temporal_layers: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    forecast_loss_weight: float
    decision_loss_weight: float
    risk_loss_weight: float


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    network: NetworkConfig
    problem: ProblemConfig
    teacher: TeacherConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    cwd = Path.cwd()

    battery = BatteryConfig(**raw["network"]["battery"])
    network = NetworkConfig(
        dss_master=(cwd / raw["network"]["dss_master"]).resolve()
        if not Path(raw["network"]["dss_master"]).is_absolute()
        else Path(raw["network"]["dss_master"]),
        pv_systems=tuple(raw["network"]["pv_systems"]),
        load_buses=tuple(str(bus) for bus in raw["network"]["load_buses"]),
        battery=battery,
    )

    return ExperimentConfig(
        seed=int(raw["seed"]),
        network=network,
        problem=ProblemConfig(**raw["problem"]),
        teacher=TeacherConfig(**raw["teacher"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
    )


def asdict_shallow(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "network": config.network,
        "problem": config.problem,
        "teacher": config.teacher,
        "model": config.model,
        "training": config.training,
    }
