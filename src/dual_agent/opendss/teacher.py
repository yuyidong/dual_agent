from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dual_agent.config import BatteryConfig, TeacherConfig
from dual_agent.opendss.ieee13_runner import OpenDSS13Runner


@dataclass(frozen=True)
class TeacherLabel:
    dispatch_kw: np.ndarray
    expected_cost: float
    voltage_risk: float
    thermal_risk: float


class RandomSearchStochasticTeacher:
    """Stochastic MPC teacher that uses OpenDSS as the power-flow evaluator."""

    def __init__(
        self,
        runner: OpenDSS13Runner,
        battery: BatteryConfig,
        teacher: TeacherConfig,
        interval_hours: float,
        rng: np.random.Generator,
    ) -> None:
        self.runner = runner
        self.battery = battery
        self.teacher = teacher
        self.interval_hours = interval_hours
        self.rng = rng

    def solve(self, pv_scenarios_kw: np.ndarray) -> TeacherLabel:
        scenarios, horizon, _nodes = pv_scenarios_kw.shape
        mean = np.zeros(horizon)
        std = np.full(horizon, self.battery.kw_rated * 0.35)

        best_schedule: np.ndarray | None = None
        best_cost = np.inf
        best_voltage = np.inf
        best_thermal = np.inf

        for _ in range(self.teacher.cem_iterations):
            candidates = self.rng.normal(mean[None, :], std[None, :], size=(self.teacher.candidates, horizon))
            candidates = np.clip(candidates, -self.battery.kw_rated, self.battery.kw_rated)

            scores = np.empty(self.teacher.candidates, dtype=float)
            voltage_risks = np.empty(self.teacher.candidates, dtype=float)
            thermal_risks = np.empty(self.teacher.candidates, dtype=float)
            for i, schedule in enumerate(candidates):
                cost, voltage, thermal = self._evaluate_schedule(schedule, pv_scenarios_kw)
                scores[i] = cost
                voltage_risks[i] = voltage
                thermal_risks[i] = thermal

            elite_count = max(1, int(self.teacher.elite_fraction * self.teacher.candidates))
            elite_idx = np.argsort(scores)[:elite_count]
            elite = candidates[elite_idx]
            mean = elite.mean(axis=0)
            std = elite.std(axis=0).clip(min=self.battery.kw_rated * 0.03)

            if scores[elite_idx[0]] < best_cost:
                best_cost = float(scores[elite_idx[0]])
                best_schedule = candidates[elite_idx[0]]
                best_voltage = float(voltage_risks[elite_idx[0]])
                best_thermal = float(thermal_risks[elite_idx[0]])

        if best_schedule is None:
            raise RuntimeError("Teacher failed to evaluate any candidate schedule.")
        return TeacherLabel(best_schedule.astype(np.float32), best_cost, best_voltage, best_thermal)

    def _evaluate_schedule(self, schedule_kw: np.ndarray, pv_scenarios_kw: np.ndarray) -> tuple[float, float, float]:
        total_cost = 0.0
        voltage_risk = 0.0
        thermal_risk = 0.0

        for scenario in pv_scenarios_kw:
            soc = self.battery.soc_initial
            scenario_cost = 0.0
            scenario_voltage = 0.0
            scenario_thermal = 0.0
            self.runner.compile()

            for t, battery_kw in enumerate(schedule_kw):
                soc = self._next_soc(soc, battery_kw)
                result = self.runner.solve_step(scenario[t], battery_kw, soc, self.teacher)
                energy_kwh = max(float(battery_kw), 0.0) * self.interval_hours
                degradation = abs(float(battery_kw)) * self.interval_hours * self.teacher.degradation_per_kwh
                voltage_penalty = self.teacher.voltage_violation_weight * result.voltage_violation
                scenario_cost += (
                    self.teacher.energy_price_per_kwh * energy_kwh
                    + degradation
                    + result.total_losses_kw * self.interval_hours * self.teacher.energy_price_per_kwh
                    + voltage_penalty
                )
                scenario_voltage += result.voltage_violation
                scenario_thermal += result.thermal_violation

            total_cost += scenario_cost / len(pv_scenarios_kw)
            voltage_risk += scenario_voltage / (len(pv_scenarios_kw) * len(schedule_kw))
            thermal_risk += scenario_thermal / (len(pv_scenarios_kw) * len(schedule_kw))

        return total_cost, voltage_risk, thermal_risk

    def _next_soc(self, soc: float, battery_kw: float) -> float:
        if battery_kw >= 0.0:
            delta = battery_kw * self.interval_hours / (self.battery.kwh_rated * self.battery.discharge_efficiency)
            soc -= delta
        else:
            delta = -battery_kw * self.interval_hours * self.battery.charge_efficiency / self.battery.kwh_rated
            soc += delta
        return float(np.clip(soc, self.battery.soc_min, self.battery.soc_max))

