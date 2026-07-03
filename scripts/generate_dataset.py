from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

import numpy as np
from tqdm import tqdm

from dual_agent.config import load_config
from dual_agent.opendss.ieee13_runner import OpenDSS13Runner
from dual_agent.opendss.teacher import RandomSearchStochasticTeacher


def synthetic_pv_scenarios(
    rng: np.random.Generator,
    samples: int,
    scenarios: int,
    history: int,
    horizon: int,
    nodes: int,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clock = np.linspace(-1.0, 1.0, history + horizon)
    clear_sky = np.maximum(0.0, 1.0 - clock**2)
    capacity = rng.uniform(150.0, 550.0, size=(nodes,))

    pv_history = np.zeros((samples, history, nodes, feature_dim), dtype=np.float32)
    pv_scenarios = np.zeros((samples, scenarios, horizon, nodes), dtype=np.float32)
    pv_target = np.zeros((samples, horizon, nodes), dtype=np.float32)

    for n in range(samples):
        cloud = rng.uniform(0.55, 1.05)
        ramp = rng.normal(0.0, 0.08, size=(history + horizon, 1)).cumsum(axis=0)
        base = clear_sky[:, None] * capacity[None, :] * np.clip(cloud + ramp, 0.1, 1.2)
        noise = rng.normal(0.0, 0.03, size=base.shape) * capacity[None, :]
        trajectory = np.maximum(base + noise, 0.0)

        hist_values = trajectory[:history]
        pv_history[n, :, :, 0] = hist_values
        pv_history[n, :, :, 1] = clear_sky[:history, None]
        pv_history[n, :, :, 2] = cloud
        pv_history[n, :, :, 3] = np.linspace(0.0, 1.0, history)[:, None]
        pv_history[n, :, :, 4] = capacity[None, :] / 1000.0
        pv_history[n, :, :, 5:] = rng.normal(0.0, 0.01, size=(history, nodes, max(0, feature_dim - 5)))

        target = trajectory[history:]
        pv_target[n] = target
        for s in range(scenarios):
            scenario_noise = rng.normal(0.0, 0.08, size=target.shape) * capacity[None, :]
            pv_scenarios[n, s] = np.maximum(target + scenario_noise, 0.0)
    return pv_history, pv_scenarios, pv_target


def synthetic_load_forecast(
    rng: np.random.Generator,
    samples: int,
    horizon: int,
    nodes: int,
    feature_dim: int,
) -> np.ndarray:
    base = rng.uniform(80.0, 350.0, size=(samples, 1, nodes, 1))
    shape = 0.8 + 0.2 * np.sin(np.linspace(0.0, 2.0 * np.pi, horizon))[None, :, None, None]
    load = base * shape
    features = [load.astype(np.float32)]
    for _ in range(feature_dim - 1):
        features.append(rng.normal(0.0, 0.05, size=load.shape).astype(np.float32))
    return np.concatenate(features, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ieee13.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()

    config = load_config(args.config)
    rng = np.random.default_rng(config.seed)
    pv_nodes = len(config.network.pv_systems)

    pv_history, pv_scenarios, pv_target = synthetic_pv_scenarios(
        rng,
        args.samples,
        config.problem.scenarios,
        config.problem.history_steps,
        config.problem.horizon_steps,
        pv_nodes,
        config.problem.pv_feature_dim,
    )
    load_forecast = synthetic_load_forecast(
        rng,
        args.samples,
        config.problem.horizon_steps,
        pv_nodes,
        config.problem.load_feature_dim,
    )

    runner = OpenDSS13Runner(config.network.dss_master, config.network.pv_systems, config.network.battery)
    teacher = RandomSearchStochasticTeacher(
        runner,
        config.network.battery,
        config.teacher,
        config.problem.interval_hours,
        rng,
    )

    dispatch = np.zeros((args.samples, config.problem.horizon_steps, 1), dtype=np.float32)
    cost = np.zeros((args.samples,), dtype=np.float32)
    voltage_risk = np.zeros((args.samples,), dtype=np.float32)
    thermal_risk = np.zeros((args.samples,), dtype=np.float32)

    for i in tqdm(range(args.samples), desc="OpenDSS teacher"):
        label = teacher.solve(pv_scenarios[i])
        dispatch[i, :, 0] = label.dispatch_kw
        cost[i] = label.expected_cost
        voltage_risk[i] = label.voltage_risk
        thermal_risk[i] = label.thermal_risk

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        pv_history=pv_history,
        pv_scenarios=pv_scenarios,
        pv_target=pv_target,
        load_forecast=load_forecast,
        dispatch=dispatch,
        cost=cost,
        voltage_risk=voltage_risk,
        thermal_risk=thermal_risk,
    )


if __name__ == "__main__":
    main()
