from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dual_agent.config import BatteryConfig, LinDistFlowConfig


@dataclass(frozen=True)
class PowerFlowResult:
    voltage_min: float
    voltage_max: float
    voltage_violation: float
    thermal_violation: float
    total_losses_kw: float


class OpenDSS13Runner:
    """Small adapter around OpenDSSDirect for IEEE 13-bus experiments."""

    def __init__(
        self,
        master_file: str | Path,
        pv_systems: tuple[str, ...],
        battery: BatteryConfig,
    ) -> None:
        try:
            import opendssdirect as dss  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenDSS support requires opendssdirect.py. Install with "
                '`pip install -e ".[opendss]"` or `pip install opendssdirect.py`.'
            ) from exc

        self.dss = dss
        self.master_file = Path(master_file)
        self.pv_systems = pv_systems
        self.battery = battery

        if not self.master_file.exists():
            raise FileNotFoundError(
                f"Cannot find OpenDSS master file: {self.master_file}. "
                "Place the IEEE 13-bus model there or update configs/ieee13.yaml."
            )

    def compile(self) -> None:
        self.dss.Basic.ClearAll()
        self.dss.Text.Command(f"compile [{self.master_file}]")
        self._ensure_battery()

    def _ensure_battery(self) -> None:
        bat = self.battery
        self.dss.Text.Command(
            "New Storage.{name} phases={phases} bus1={bus} kv={kv} "
            "kwrated={kw} kwhrated={kwh} %stored={soc} %reserve=0 "
            "state=idling".format(
                name=bat.name,
                phases=bat.phases,
                bus=bat.bus,
                kv=bat.kv,
                kw=bat.kw_rated,
                kwh=bat.kwh_rated,
                soc=100.0 * bat.soc_initial,
            )
        )

    def set_pv_kw(self, pv_kw: np.ndarray) -> None:
        if len(pv_kw) != len(self.pv_systems):
            raise ValueError("pv_kw length must match configured pv_systems")
        for name, kw in zip(self.pv_systems, pv_kw, strict=True):
            self.dss.Text.Command(f"Edit PVSystem.{name} pmpp={max(float(kw), 0.0):.6f}")

    def set_battery_kw(self, kw: float, soc: float) -> None:
        state = "discharging" if kw >= 0 else "charging"
        self.dss.Text.Command(
            f"Edit Storage.{self.battery.name} state={state} kw={abs(float(kw)):.6f} "
            f"%stored={100.0 * float(soc):.6f}"
        )

    def solve_step(
        self,
        pv_kw: np.ndarray,
        battery_kw: float,
        soc: float,
        lindistflow: LinDistFlowConfig,
    ) -> PowerFlowResult:
        self.set_pv_kw(pv_kw)
        self.set_battery_kw(battery_kw, soc)
        self.dss.Solution.Solve()

        voltages = np.asarray(self.dss.Circuit.AllBusMagPu(), dtype=float)
        voltage_min = float(np.min(voltages))
        voltage_max = float(np.max(voltages))
        low = np.maximum(lindistflow.voltage_lower - voltages, 0.0)
        high = np.maximum(voltages - lindistflow.voltage_upper, 0.0)
        voltage_violation = float(np.mean(low**2 + high**2))

        thermal_violation = 0.0
        try:
            powers = np.asarray(self.dss.CktElement.Powers(), dtype=float)
            thermal_violation = float(np.mean(np.maximum(np.abs(powers) - 1e9, 0.0)))
        except Exception:
            thermal_violation = 0.0

        losses_kw = float(self.dss.Circuit.Losses()[0] / 1000.0)
        return PowerFlowResult(
            voltage_min=voltage_min,
            voltage_max=voltage_max,
            voltage_violation=voltage_violation,
            thermal_violation=thermal_violation,
            total_losses_kw=losses_kw,
        )
