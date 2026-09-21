from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

import numpy as np

from dual_agent.config import load_config


def synthetic_pv_scenarios(
    rng: np.random.Generator,
    samples: int,
    scenarios: int,
    history: int,
    horizon: int,
    interval_hours: float,
    nodes: int,
    feature_dim: int,
    generate_scenarios: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    time_steps = np.arange(-history, horizon)
    time_of_day = (time_steps * interval_hours) % 24.0
    capacity = rng.uniform(150.0, 550.0, size=(nodes,))
    node_clear_sky_scale = np.linspace(0.92, 1.08, nodes)
    node_sun_shift = np.linspace(-0.45, 0.45, nodes)
    station_risk = np.linspace(0.0, 1.0, nodes)

    pv_history = np.zeros((samples, history, nodes, feature_dim), dtype=np.float32)
    pv_scenarios = (
        np.zeros((samples, scenarios, horizon, nodes), dtype=np.float32)
        if generate_scenarios
        else None
    )
    pv_target = np.zeros((samples, horizon, nodes), dtype=np.float32)

    for n in range(samples):
        season = 0.86 + 0.14 * np.sin(2.0 * np.pi * (n % 365) / 365.0)
        sunrise_shift = rng.normal(0.0, 0.35)
        node_day_shift = sunrise_shift + node_sun_shift + rng.normal(0.0, 0.12, size=(nodes,))
        seasonal_phase = np.pi * (time_of_day[:, None] - 6.0 - node_day_shift[None, :]) / 12.0
        sample_clear_sky = (
            np.maximum(0.0, np.sin(seasonal_phase))
            * season
            * node_clear_sky_scale[None, :]
        )

        weather_type = rng.choice(4, p=[0.42, 0.28, 0.20, 0.10])
        cloud_base, cloud_volatility, event_rate = (
            (0.96, 0.025, 0.15)
            if weather_type == 0
            else (0.78, 0.070, 0.65)
            if weather_type == 1
            else (0.58, 0.110, 1.20)
            if weather_type == 2
            else (0.34, 0.060, 0.35)
        )
        cloud = _cloud_transmissivity(
            rng,
            history + horizon,
            cloud_base,
            cloud_volatility,
            event_rate,
            horizon,
        )
        local_cloud = _station_cloud_transmissivity(
            rng,
            history + horizon,
            nodes,
            horizon,
            time_of_day,
            weather_type,
        )
        station_cloud = np.clip(0.58 * cloud[:, None] + 0.42 * local_cloud, 0.03, 1.15)
        node_cloud_bias = rng.normal(1.0, 0.035, size=(nodes,))
        base = sample_clear_sky * capacity[None, :] * np.clip(station_cloud * node_cloud_bias, 0.03, 1.15)
        noise = (
            rng.normal(0.0, 0.045, size=base.shape)
            * capacity[None, :]
            * sample_clear_sky
            * np.clip(1.35 - station_cloud, 0.35, 1.35)
        )
        trajectory = np.maximum(base + noise, 0.0)

        hist_values = trajectory[:history]
        noisy_cloud_estimate = np.clip(
            station_cloud[:history] + rng.normal(0.0, 0.11, size=(history, nodes)),
            0.0,
            1.2,
        )
        pv_history[n, :, :, 0] = hist_values
        pv_history[n, :, :, 1] = sample_clear_sky[:history]
        pv_history[n, :, :, 2] = noisy_cloud_estimate
        pv_history[n, :, :, 3] = time_of_day[:history, None] / 24.0
        pv_history[n, :, :, 4] = capacity[None, :] / 1000.0
        if feature_dim > 5:
            weather_feature = (
                0.70 * weather_type / 3.0
                + 0.30 * station_risk[None, :, None]
                + rng.normal(0.0, 0.08, size=(history, nodes, 1))
            )
            pv_history[n, :, :, 5:6] = np.clip(weather_feature, 0.0, 1.0)
        if feature_dim > 6:
            pv_history[n, :, :, 6:] = rng.normal(0.0, 0.05, size=(history, nodes, feature_dim - 6))

        target = trajectory[history:]
        pv_target[n] = target
        if pv_scenarios is not None:
            for s in range(scenarios):
                scenario_cloud = np.clip(
                    cloud[history:] + rng.normal(0.0, 0.12, size=horizon).cumsum() / max(2.0, horizon**0.5),
                    0.02,
                    1.15,
                )
                scenario_local_cloud = _station_cloud_transmissivity(
                    rng,
                    horizon,
                    nodes,
                    horizon,
                    time_of_day[history:],
                    weather_type,
                )
                scenario_station_cloud = np.clip(
                    0.58 * scenario_cloud[:, None] + 0.42 * scenario_local_cloud,
                    0.03,
                    1.15,
                )
                scenario_base = sample_clear_sky[history:] * capacity[None, :] * scenario_station_cloud
                scenario_noise = (
                    rng.normal(0.0, 0.07, size=target.shape)
                    * capacity[None, :]
                    * sample_clear_sky[history:]
                )
                pv_scenarios[n, s] = np.maximum(scenario_base + scenario_noise, 0.0)
    return pv_history, pv_scenarios, pv_target


def _cloud_transmissivity(
    rng: np.random.Generator,
    steps: int,
    base: float,
    volatility: float,
    event_rate: float,
    horizon: int,
) -> np.ndarray:
    cloud = np.empty(steps, dtype=np.float32)
    cloud[0] = np.clip(base + rng.normal(0.0, volatility), 0.03, 1.15)
    for idx in range(1, steps):
        cloud[idx] = 0.88 * cloud[idx - 1] + 0.12 * base + rng.normal(0.0, volatility)

    event_count = rng.poisson(event_rate)
    future_start = max(0, steps - horizon)
    for _ in range(event_count):
        center = rng.integers(future_start, steps)
        width = rng.uniform(1.0, max(2.0, 0.18 * horizon))
        severity = rng.uniform(0.18, 0.72)
        event_shape = np.exp(-0.5 * ((np.arange(steps) - center) / width) ** 2)
        cloud -= severity * event_shape

    return np.clip(cloud, 0.03, 1.15)


def _station_cloud_transmissivity(
    rng: np.random.Generator,
    steps: int,
    nodes: int,
    horizon: int,
    time_of_day: np.ndarray,
    weather_type: int,
) -> np.ndarray:
    local = np.empty((steps, nodes), dtype=np.float32)
    station_base = np.array([0.98, 0.90, 0.94, 0.86], dtype=np.float32)
    station_volatility = np.array([0.018, 0.055, 0.040, 0.075], dtype=np.float32)
    station_event_rate = np.array([0.10, 0.85, 0.55, 1.05], dtype=np.float32)
    station_event_center = np.array([11.0, 12.5, 15.0, 10.5], dtype=np.float32)
    station_event_width = np.array([2.6, 1.3, 2.1, 1.1], dtype=np.float32)

    weather_event_scale = (0.25, 0.75, 1.35, 0.55)[weather_type]
    weather_volatility_scale = (0.75, 1.0, 1.25, 0.65)[weather_type]
    index = np.arange(steps)
    future_start = max(0, steps - horizon)

    for node in range(nodes):
        profile = node % len(station_base)
        base = station_base[profile] + rng.normal(0.0, 0.025)
        volatility = station_volatility[profile] * weather_volatility_scale
        series = np.empty(steps, dtype=np.float32)
        series[0] = np.clip(base + rng.normal(0.0, volatility), 0.03, 1.15)
        for idx in range(1, steps):
            series[idx] = 0.82 * series[idx - 1] + 0.18 * base + rng.normal(0.0, volatility)

        event_count = rng.poisson(station_event_rate[profile] * weather_event_scale)
        for _ in range(event_count):
            if rng.random() < 0.70:
                event_time = station_event_center[profile] + rng.normal(0.0, station_event_width[profile])
                candidate = np.where(np.abs(time_of_day - event_time) <= 4.5)[0]
                center = int(rng.choice(candidate)) if candidate.size else int(rng.integers(0, steps))
            else:
                center = int(rng.integers(future_start, steps))
            width = rng.uniform(0.8, max(1.2, 0.14 * horizon))
            severity = rng.uniform(0.15, 0.55 + 0.10 * profile)
            event_shape = np.exp(-0.5 * ((index - center) / width) ** 2)
            series -= severity * event_shape

        local[:, node] = np.clip(series, 0.03, 1.15)

    return local


def synthetic_load_forecast(
    rng: np.random.Generator,
    samples: int,
    horizon: int,
    interval_hours: float,
    nodes: int,
    feature_dim: int,
) -> np.ndarray:
    base = rng.uniform(80.0, 350.0, size=(samples, 1, nodes, 1))
    time_of_day = (np.arange(horizon) * interval_hours) % 24.0
    midday_peak = np.exp(-0.5 * ((time_of_day - 12.0) / 3.0) ** 2)
    evening_peak = np.exp(-0.5 * ((time_of_day - 19.0) / 2.5) ** 2)
    morning_ramp = np.exp(-0.5 * ((time_of_day - 8.0) / 3.5) ** 2)
    shape = 0.62 + 0.18 * morning_ramp + 0.28 * midday_peak + 0.38 * evening_peak
    shape = shape / shape.mean()
    shape = shape[None, :, None, None]
    load = base * shape
    features = [load.astype(np.float32)]
    for _ in range(feature_dim - 1):
        features.append(rng.normal(0.0, 0.05, size=load.shape).astype(np.float32))
    return np.concatenate(features, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ieee13.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=None, help="Override problem.samples from the config.")
    args = parser.parse_args()

    config = load_config(args.config)
    samples = config.problem.samples if args.samples is None else args.samples
    rng = np.random.default_rng(config.seed)
    pv_nodes = len(config.network.pv_systems)

    pv_history, pv_scenarios, pv_target = synthetic_pv_scenarios(
        rng,
        samples,
        config.problem.scenarios,
        config.problem.history_steps,
        config.problem.horizon_steps,
        config.problem.interval_hours,
        pv_nodes,
        config.problem.pv_feature_dim,
        config.problem.generate_dataset_scenarios,
    )
    load_forecast = synthetic_load_forecast(
        rng,
        samples,
        config.problem.horizon_steps,
        config.problem.interval_hours,
        pv_nodes,
        config.problem.load_feature_dim,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "pv_history": pv_history,
        "pv_target": pv_target,
        "load_forecast": load_forecast,
    }
    if pv_scenarios is not None:
        arrays["pv_scenarios"] = pv_scenarios
    np.savez_compressed(
        out,
        **arrays,
    )


if __name__ == "__main__":
    main()
