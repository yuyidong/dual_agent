from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dual_agent.config import BatteryConfig, LinDistFlowConfig


@dataclass(frozen=True)
class TeacherLabel:
    dispatch_kw: np.ndarray
    expected_cost: float
    voltage_risk: float
    thermal_risk: float


@dataclass(frozen=True)
class Branch:
    parent: str
    child: str
    r_pu: float
    x_pu: float
    limit_mw: float


@dataclass(frozen=True)
class LinDistFlowNetwork:
    root_bus: str
    buses: tuple[str, ...]
    branches: tuple[Branch, ...]
    base_load_kw: dict[str, float]
    base_load_kvar: dict[str, float]


class LinDistFlowStochasticTeacher:
    """Stochastic OPF teacher using a balanced LinDistFlow LP."""

    def __init__(
        self,
        master_file: str | Path,
        pv_systems: tuple[str, ...],
        battery: BatteryConfig,
        teacher: LinDistFlowConfig,
        interval_hours: float,
    ) -> None:
        self.network = parse_ieee13_lindistflow(master_file)
        self.pv_buses = tuple(_name_to_bus(name) for name in pv_systems)
        self.battery = battery
        self.teacher = teacher
        self.interval_hours = interval_hours
        self.battery_bus = _bus_name(battery.bus)

        if self.battery_bus not in self.network.buses:
            raise ValueError(f"Battery bus {self.battery_bus} is not present in the LinDistFlow network.")
        missing_pv = sorted(set(self.pv_buses) - set(self.network.buses))
        if missing_pv:
            raise ValueError(f"PV buses are not present in the LinDistFlow network: {missing_pv}")

    def solve(self, pv_scenarios_kw: np.ndarray, load_forecast: np.ndarray | None = None) -> TeacherLabel:
        try:
            import pyomo.environ as pyo
            from pyomo.opt import TerminationCondition
        except ImportError as exc:
            raise ImportError(
                "The LinDistFlow OPF teacher requires Pyomo. Install with `pip install pyomo highspy`."
            ) from exc

        scenarios, horizon, pv_nodes = pv_scenarios_kw.shape
        if pv_nodes != len(self.pv_buses):
            raise ValueError("pv_scenarios_kw node dimension must match configured PV systems.")

        buses = self.network.buses
        branches = self.network.branches
        non_root = tuple(bus for bus in buses if bus != self.network.root_bus)
        bus_index = {bus: i for i, bus in enumerate(buses)}
        branch_index = {(branch.parent, branch.child): i for i, branch in enumerate(branches)}
        incoming = {branch.child: branch_index[(branch.parent, branch.child)] for branch in branches}
        outgoing: dict[str, list[int]] = {bus: [] for bus in buses}
        for e, branch in enumerate(branches):
            outgoing[branch.parent].append(e)

        load_kw, load_kvar = self._expand_load(load_forecast, horizon)
        p_batt_max_mw = self.battery.kw_rated / 1000.0
        v_lower = self.teacher.voltage_lower**2
        v_upper = self.teacher.voltage_upper**2
        soc_discharge_coeff = 1000.0 * self.interval_hours / (
            self.battery.kwh_rated * self.battery.discharge_efficiency
        )
        soc_charge_coeff = 1000.0 * self.interval_hours * self.battery.charge_efficiency / self.battery.kwh_rated

        model = pyo.ConcreteModel(name="stochastic_lindistflow_opf")
        model.S = pyo.RangeSet(0, scenarios - 1)
        model.T = pyo.RangeSet(0, horizon - 1)
        model.B = pyo.RangeSet(0, len(buses) - 1)
        model.E = pyo.RangeSet(0, len(branches) - 1)
        model.NR = pyo.Set(initialize=[bus_index[bus] for bus in non_root])

        model.p_dis = pyo.Var(model.T, bounds=(0.0, p_batt_max_mw))
        model.p_ch = pyo.Var(model.T, bounds=(0.0, p_batt_max_mw))
        model.soc = pyo.Var(model.T, bounds=(self.battery.soc_min, self.battery.soc_max))
        model.p_flow = pyo.Var(model.S, model.T, model.E)
        model.q_flow = pyo.Var(model.S, model.T, model.E)
        model.v = pyo.Var(model.S, model.T, model.B)
        model.v_low = pyo.Var(model.S, model.T, model.B, within=pyo.NonNegativeReals)
        model.v_high = pyo.Var(model.S, model.T, model.B, within=pyo.NonNegativeReals)
        model.thermal = pyo.Var(model.S, model.T, model.E, within=pyo.NonNegativeReals)

        root_edges = tuple(outgoing[self.network.root_bus])
        battery_bus_idx = bus_index[self.battery_bus]
        branch_parent_idx = {e: bus_index[branch.parent] for e, branch in enumerate(branches)}
        branch_child_idx = {e: bus_index[branch.child] for e, branch in enumerate(branches)}
        branch_r = {e: branch.r_pu for e, branch in enumerate(branches)}
        branch_x = {e: branch.x_pu for e, branch in enumerate(branches)}
        branch_limit = {e: branch.limit_mw for e, branch in enumerate(branches)}
        incoming_by_bus_idx = {bus_index[bus]: edge for bus, edge in incoming.items()}
        outgoing_by_bus_idx = {bus_index[bus]: tuple(edges) for bus, edges in outgoing.items()}
        net_load_mw = {
            (s, t, bus_index[bus]): self._net_load_mw(bus, t, s, load_kw, pv_scenarios_kw)
            for s in range(scenarios)
            for t in range(horizon)
            for bus in non_root
        }
        reactive_load_mvar = {
            (t, b): float(load_kvar[t, b]) / 1000.0 for t in range(horizon) for b in range(len(buses))
        }

        def soc_rule(m: pyo.ConcreteModel, t: int) -> pyo.Expression:
            previous_soc = self.battery.soc_initial if t == 0 else m.soc[t - 1]
            return (
                m.soc[t]
                == previous_soc - soc_discharge_coeff * m.p_dis[t] + soc_charge_coeff * m.p_ch[t]
            )

        model.soc_dynamics = pyo.Constraint(model.T, rule=soc_rule)

        def active_balance_rule(m: pyo.ConcreteModel, s: int, t: int, b: int) -> pyo.Expression:
            edge_in = incoming_by_bus_idx[b]
            battery_injection = m.p_dis[t] - m.p_ch[t] if b == battery_bus_idx else 0.0
            return (
                m.p_flow[s, t, edge_in]
                - sum(m.p_flow[s, t, edge] for edge in outgoing_by_bus_idx[b])
                + battery_injection
                == net_load_mw[s, t, b]
            )

        model.active_balance = pyo.Constraint(model.S, model.T, model.NR, rule=active_balance_rule)

        def reactive_balance_rule(m: pyo.ConcreteModel, s: int, t: int, b: int) -> pyo.Expression:
            edge_in = incoming_by_bus_idx[b]
            return (
                m.q_flow[s, t, edge_in]
                - sum(m.q_flow[s, t, edge] for edge in outgoing_by_bus_idx[b])
                == reactive_load_mvar[t, b]
            )

        model.reactive_balance = pyo.Constraint(model.S, model.T, model.NR, rule=reactive_balance_rule)

        def root_voltage_rule(m: pyo.ConcreteModel, s: int, t: int) -> pyo.Expression:
            return m.v[s, t, bus_index[self.network.root_bus]] == 1.0

        model.root_voltage = pyo.Constraint(model.S, model.T, rule=root_voltage_rule)

        def voltage_drop_rule(m: pyo.ConcreteModel, s: int, t: int, e: int) -> pyo.Expression:
            return (
                m.v[s, t, branch_child_idx[e]]
                == m.v[s, t, branch_parent_idx[e]]
                - 2.0 * (branch_r[e] * m.p_flow[s, t, e] + branch_x[e] * m.q_flow[s, t, e])
            )

        model.voltage_drop = pyo.Constraint(model.S, model.T, model.E, rule=voltage_drop_rule)

        def voltage_lower_rule(m: pyo.ConcreteModel, s: int, t: int, b: int) -> pyo.Expression:
            return m.v[s, t, b] + m.v_low[s, t, b] >= v_lower

        def voltage_upper_rule(m: pyo.ConcreteModel, s: int, t: int, b: int) -> pyo.Expression:
            return m.v[s, t, b] - m.v_high[s, t, b] <= v_upper

        model.voltage_lower = pyo.Constraint(model.S, model.T, model.B, rule=voltage_lower_rule)
        model.voltage_upper = pyo.Constraint(model.S, model.T, model.B, rule=voltage_upper_rule)

        def thermal_upper_rule(m: pyo.ConcreteModel, s: int, t: int, e: int) -> pyo.Expression:
            return m.p_flow[s, t, e] - m.thermal[s, t, e] <= branch_limit[e]

        def thermal_lower_rule(m: pyo.ConcreteModel, s: int, t: int, e: int) -> pyo.Expression:
            return -m.p_flow[s, t, e] - m.thermal[s, t, e] <= branch_limit[e]

        model.thermal_upper = pyo.Constraint(model.S, model.T, model.E, rule=thermal_upper_rule)
        model.thermal_lower = pyo.Constraint(model.S, model.T, model.E, rule=thermal_lower_rule)

        def objective_rule(m: pyo.ConcreteModel) -> pyo.Expression:
            energy_cost = sum(
                self.teacher.energy_price_per_kwh
                * 1000.0
                * self.interval_hours
                * m.p_flow[s, t, edge]
                / scenarios
                for s in m.S
                for t in m.T
                for edge in root_edges
            )
            degradation = sum(
                self.teacher.degradation_per_kwh * 1000.0 * self.interval_hours * (m.p_dis[t] + m.p_ch[t])
                for t in m.T
            )
            voltage_penalty = sum(
                (m.v_low[s, t, b] + m.v_high[s, t, b]) / (scenarios * horizon)
                for s in m.S
                for t in m.T
                for b in m.B
            )
            thermal_penalty = sum(
                m.thermal[s, t, e] / (scenarios * horizon)
                for s in m.S
                for t in m.T
                for e in m.E
            )
            return energy_cost + degradation + voltage_penalty + thermal_penalty

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        solver = pyo.SolverFactory("appsi_highs")
        if not solver.available(False):
            solver = pyo.SolverFactory("highs")
        if not solver.available(False):
            raise RuntimeError("No Pyomo HiGHS solver is available. Install with `pip install highspy`.")
        result = solver.solve(model, tee=False)
        termination = result.solver.termination_condition
        if termination not in {TerminationCondition.optimal, TerminationCondition.feasible}:
            raise RuntimeError(f"LinDistFlow Pyomo OPF failed: {termination}")

        dispatch_kw = np.array(
            [1000.0 * (pyo.value(model.p_dis[t]) - pyo.value(model.p_ch[t])) for t in range(horizon)],
            dtype=np.float32,
        )
        voltage_risk = float(
            np.mean(
                [
                    pyo.value(model.v_low[s, t, b]) + pyo.value(model.v_high[s, t, b])
                    for s in range(scenarios)
                    for t in range(horizon)
                    for b in range(len(buses))
                ]
            )
        )
        thermal_risk = float(
            np.mean(
                [
                    pyo.value(model.thermal[s, t, e])
                    for s in range(scenarios)
                    for t in range(horizon)
                    for e in range(len(branches))
                ]
            )
        )
        return TeacherLabel(dispatch_kw, float(pyo.value(model.objective)), voltage_risk, thermal_risk)

    def _expand_load(self, load_forecast: np.ndarray | None, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        buses = self.network.buses
        base_kw = np.array([self.network.base_load_kw.get(bus, 0.0) for bus in buses], dtype=float)
        base_kvar = np.array([self.network.base_load_kvar.get(bus, 0.0) for bus in buses], dtype=float)
        base_total = float(base_kw.sum())
        if load_forecast is None:
            return np.tile(base_kw[None, :], (horizon, 1)), np.tile(base_kvar[None, :], (horizon, 1))

        load_signal = np.asarray(load_forecast, dtype=float)
        if load_signal.ndim == 3:
            total_kw = load_signal[:, :, 0].sum(axis=1)
        elif load_signal.ndim == 2:
            total_kw = load_signal[:, 0]
        else:
            raise ValueError("load_forecast must have shape (horizon, nodes, features) or (horizon, features).")

        scale = total_kw / max(base_total, 1e-6)
        return scale[:, None] * base_kw[None, :], scale[:, None] * base_kvar[None, :]

    def _net_load_mw(
        self,
        bus: str,
        t: int,
        s: int,
        load_kw: np.ndarray,
        pv_scenarios_kw: np.ndarray,
    ) -> float:
        bus_idx = self.network.buses.index(bus)
        pv_kw = 0.0
        for pv_idx, pv_bus in enumerate(self.pv_buses):
            if pv_bus == bus:
                pv_kw += float(pv_scenarios_kw[s, t, pv_idx])
        return (float(load_kw[t, bus_idx]) - pv_kw) / 1000.0


def parse_ieee13_lindistflow(master_file: str | Path) -> LinDistFlowNetwork:
    path = Path(master_file)
    text = _join_continuations(path.read_text())
    linecodes = _parse_linecodes(text)
    base_load_kw, base_load_kvar = _parse_loads(text)
    branches = _parse_branches(text, linecodes)
    buses = tuple(dict.fromkeys([bus for branch in branches for bus in (branch.parent, branch.child)]))
    return LinDistFlowNetwork("RG60", buses, tuple(branches), base_load_kw, base_load_kvar)


def _join_continuations(text: str) -> str:
    commands: list[str] = []
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.split("!")[0].strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("~"):
            current += " " + line[1:].strip()
        else:
            if current:
                commands.append(current)
            current = line
    if current:
        commands.append(current)
    return "\n".join(commands)


def _parse_linecodes(text: str) -> dict[str, tuple[float, float]]:
    linecodes: dict[str, tuple[float, float]] = {}
    pattern = re.compile(r"New\s+linecode\.(\w+).*?rmatrix\s*=\s*[\[(](.*?)[\])].*?xmatrix\s*=\s*[\[(](.*?)[\])]", re.I)
    for match in pattern.finditer(text):
        name = match.group(1).lower()
        r_values = _numbers(match.group(2))
        x_values = _numbers(match.group(3))
        if r_values and x_values:
            linecodes[name] = (float(np.mean(r_values)), float(np.mean(x_values)))
    return linecodes


def _parse_loads(text: str) -> tuple[dict[str, float], dict[str, float]]:
    kw: dict[str, float] = {}
    kvar: dict[str, float] = {}
    for line in text.splitlines():
        if not re.match(r"New\s+Load\.", line, re.I):
            continue
        bus_match = re.search(r"Bus1=([^\s]+)", line, re.I)
        kw_match = re.search(r"kW=([0-9.eE+-]+)", line, re.I)
        kvar_match = re.search(r"kvar=([0-9.eE+-]+)", line, re.I)
        if bus_match and kw_match:
            bus = _bus_name(bus_match.group(1))
            kw[bus] = kw.get(bus, 0.0) + float(kw_match.group(1))
            kvar[bus] = kvar.get(bus, 0.0) + float(kvar_match.group(1)) if kvar_match else kvar.get(bus, 0.0)
    return kw, kvar


def _parse_branches(text: str, linecodes: dict[str, tuple[float, float]]) -> list[Branch]:
    branches: list[Branch] = []
    z_base_ohm = 4.16**2
    for line in text.splitlines():
        if not re.match(r"New\s+Line\.", line, re.I):
            continue
        bus1 = re.search(r"Bus1=([^\s]+)", line, re.I)
        bus2 = re.search(r"Bus2=([^\s]+)", line, re.I)
        if not bus1 or not bus2:
            continue
        if re.search(r"Switch=y", line, re.I):
            r_ohm = 1e-4
            x_ohm = 1e-4
        else:
            code_match = re.search(r"LineCode=([^\s]+)", line, re.I)
            length_match = re.search(r"Length=([0-9.eE+-]+)", line, re.I)
            units_match = re.search(r"units=([^\s]+)", line, re.I)
            if not code_match or not length_match:
                continue
            r_per_mi, x_per_mi = linecodes[code_match.group(1).lower()]
            length = float(length_match.group(1))
            units = units_match.group(1).lower() if units_match else "mi"
            miles = length / 5280.0 if units == "ft" else length
            r_ohm = r_per_mi * miles
            x_ohm = x_per_mi * miles
        branches.append(
            Branch(
                parent=_bus_name(bus1.group(1)),
                child=_bus_name(bus2.group(1)),
                r_pu=r_ohm / z_base_ohm,
                x_pu=x_ohm / z_base_ohm,
                limit_mw=5.0,
            )
        )
    branches.append(Branch("633", "634", r_pu=0.0005, x_pu=0.0020, limit_mw=0.5))
    return branches


def _numbers(value: str) -> list[float]:
    return [float(item) for item in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", value)]


def _bus_name(bus: str) -> str:
    return bus.split(".")[0]


def _name_to_bus(name: str) -> str:
    return name.removeprefix("pv_")
