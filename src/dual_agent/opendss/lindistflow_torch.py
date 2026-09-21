from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from dual_agent.config import BatteryConfig, LinDistFlowConfig
from dual_agent.opendss.lindistflow_network import _bus_name, _name_to_bus, parse_ieee13_lindistflow


class TorchLinDistFlowEvaluator(nn.Module):
    """Differentiable LinDistFlow evaluator for a fixed battery dispatch."""

    def __init__(
        self,
        master_file: str | Path,
        pv_systems: tuple[str, ...],
        battery: BatteryConfig | tuple[BatteryConfig, ...],
        lindistflow: LinDistFlowConfig,
        interval_hours: float,
    ) -> None:
        super().__init__()
        network = parse_ieee13_lindistflow(master_file)
        buses = network.buses
        bus_index = {bus: i for i, bus in enumerate(buses)}
        pv_buses = tuple(_name_to_bus(name) for name in pv_systems)
        batteries = _as_battery_tuple(battery)
        if not batteries:
            raise ValueError("At least one battery must be configured.")
        battery_buses = tuple(_bus_name(item.bus) for item in batteries)

        missing_batteries = sorted(set(battery_buses) - set(buses))
        if missing_batteries:
            raise ValueError(f"Battery buses are not present in the LinDistFlow network: {missing_batteries}")
        missing_pv = sorted(set(pv_buses) - set(buses))
        if missing_pv:
            raise ValueError(f"PV buses are not present in the LinDistFlow network: {missing_pv}")

        parent_idx = [bus_index[branch.parent] for branch in network.branches]
        child_idx = [bus_index[branch.child] for branch in network.branches]
        root_idx = bus_index[network.root_bus]
        edge_order = _topological_edge_order(parent_idx, child_idx, root_idx)
        pv_bus_matrix = torch.zeros(len(pv_buses), len(buses))
        for pv_idx, bus in enumerate(pv_buses):
            pv_bus_matrix[pv_idx, bus_index[bus]] = 1.0

        base_kw = torch.tensor([network.base_load_kw.get(bus, 0.0) for bus in buses], dtype=torch.float32)
        base_kvar = torch.tensor([network.base_load_kvar.get(bus, 0.0) for bus in buses], dtype=torch.float32)
        root_edges = [idx for idx, parent in enumerate(parent_idx) if parent == bus_index[network.root_bus]]

        self.register_buffer("parent_idx", torch.tensor(parent_idx, dtype=torch.long))
        self.register_buffer("child_idx", torch.tensor(child_idx, dtype=torch.long))
        self.register_buffer("edge_order", torch.tensor(edge_order, dtype=torch.long))
        self.register_buffer("branch_r", torch.tensor([branch.r_pu for branch in network.branches], dtype=torch.float32))
        self.register_buffer("branch_x", torch.tensor([branch.x_pu for branch in network.branches], dtype=torch.float32))
        self.register_buffer(
            "branch_limit",
            torch.tensor([branch.limit_mw for branch in network.branches], dtype=torch.float32),
        )
        self.register_buffer("root_edges", torch.tensor(root_edges, dtype=torch.long))
        self.register_buffer("base_kw", base_kw)
        self.register_buffer("base_kvar", base_kvar)
        self.register_buffer("pv_bus_matrix", pv_bus_matrix)
        self.register_buffer("battery_bus_idx", torch.tensor([bus_index[bus] for bus in battery_buses], dtype=torch.long))
        self.register_buffer("kw_rated", torch.tensor([item.kw_rated for item in batteries], dtype=torch.float32))
        self.register_buffer("kwh_rated", torch.tensor([item.kwh_rated for item in batteries], dtype=torch.float32))
        self.register_buffer("soc_initial", torch.tensor([item.soc_initial for item in batteries], dtype=torch.float32))
        self.register_buffer("soc_min", torch.tensor([item.soc_min for item in batteries], dtype=torch.float32))
        self.register_buffer("soc_max", torch.tensor([item.soc_max for item in batteries], dtype=torch.float32))
        self.register_buffer(
            "charge_efficiency", torch.tensor([item.charge_efficiency for item in batteries], dtype=torch.float32)
        )
        self.register_buffer(
            "discharge_efficiency",
            torch.tensor([item.discharge_efficiency for item in batteries], dtype=torch.float32),
        )

        self.bus_count = len(buses)
        self.branch_count = len(network.branches)
        self.root_idx = root_idx
        self.battery_count = len(batteries)
        self.interval_hours = interval_hours
        self.energy_price_per_kwh = lindistflow.energy_price_per_kwh
        self.energy_price_profile_per_kwh = lindistflow.energy_price_profile_per_kwh
        self.degradation_per_kwh = lindistflow.degradation_per_kwh
        self.curtailment_per_kwh = lindistflow.curtailment_per_kwh
        self.voltage_lower_sq = lindistflow.voltage_lower**2
        self.voltage_upper_sq = lindistflow.voltage_upper**2
        self.terminal_soc_tolerance = lindistflow.terminal_soc_tolerance

    def forward(
        self,
        dispatch_kw: torch.Tensor,
        pv_scenarios_kw: torch.Tensor,
        load_forecast: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if dispatch_kw.ndim == 2:
            dispatch_kw = dispatch_kw.unsqueeze(-1)
        if dispatch_kw.ndim != 3 or dispatch_kw.size(-1) != self.battery_count:
            raise ValueError("dispatch_kw must have shape [batch, horizon, battery_count]")

        batch, scenarios, horizon, _ = pv_scenarios_kw.shape
        load_kw, load_kvar = self._expand_load(load_forecast)
        pv_by_bus_kw = torch.einsum("bshn,nk->bshk", pv_scenarios_kw, self.pv_bus_matrix)
        active_accum = load_kw[:, None, :, :] / 1000.0 - pv_by_bus_kw / 1000.0
        active_accum = active_accum.clone()
        for battery_idx in range(self.battery_count):
            bus_idx = int(self.battery_bus_idx[battery_idx])
            active_accum[..., bus_idx] = active_accum[..., bus_idx] - dispatch_kw[:, None, :, battery_idx] / 1000.0

        reactive_accum = (load_kvar[:, None, :, :] / 1000.0).expand(batch, scenarios, horizon, self.bus_count).clone()
        p_flow = dispatch_kw.new_zeros(batch, scenarios, horizon, self.branch_count)
        q_flow = dispatch_kw.new_zeros(batch, scenarios, horizon, self.branch_count)
        for edge in reversed(self.edge_order.tolist()):
            child = int(self.child_idx[edge])
            parent = int(self.parent_idx[edge])
            p_flow[..., edge] = active_accum[..., child]
            q_flow[..., edge] = reactive_accum[..., child]
            active_accum[..., parent] = active_accum[..., parent] + p_flow[..., edge]
            reactive_accum[..., parent] = reactive_accum[..., parent] + q_flow[..., edge]

        voltage_by_bus: list[torch.Tensor | None] = [None] * self.bus_count
        voltage_by_bus[self.root_idx] = torch.ones(batch, scenarios, horizon, device=dispatch_kw.device)
        for edge in self.edge_order.tolist():
            parent = int(self.parent_idx[edge])
            child = int(self.child_idx[edge])
            parent_voltage = voltage_by_bus[parent]
            if parent_voltage is None:
                raise RuntimeError("LinDistFlow branches must be ordered from root to leaves.")
            voltage_by_bus[child] = parent_voltage - 2.0 * (
                self.branch_r[edge] * p_flow[..., edge] + self.branch_x[edge] * q_flow[..., edge]
            )
        if any(item is None for item in voltage_by_bus):
            raise RuntimeError("LinDistFlow network contains buses unreachable from the root.")
        voltage = torch.stack([item for item in voltage_by_bus if item is not None], dim=-1)

        voltage_low = torch.relu(self.voltage_lower_sq - voltage)
        voltage_high = torch.relu(voltage - self.voltage_upper_sq)
        voltage_violation = (voltage_low + voltage_high).mean(dim=(1, 2, 3))
        line_flow_violation = torch.relu(p_flow.abs() - self.branch_limit).mean(dim=(1, 2, 3))

        root_flow = p_flow.index_select(dim=-1, index=self.root_edges).sum(dim=-1)
        energy_price = self._energy_price(horizon, dispatch_kw.device, dispatch_kw.dtype)
        import_mw = torch.relu(root_flow)
        surplus_mw = torch.relu(-root_flow)
        energy_cost = (
            1000.0
            * self.interval_hours
            * (import_mw * energy_price[None, None, :]).sum(dim=(1, 2))
            / scenarios
        )
        curtailment = self.curtailment_per_kwh * 1000.0 * self.interval_hours * surplus_mw.sum(dim=(1, 2)) / scenarios
        degradation = self.degradation_per_kwh * self.interval_hours * dispatch_kw.abs().sum(dim=(1, 2))
        operating_cost = energy_cost + degradation + curtailment
        network_violation = voltage_violation + line_flow_violation
        objective = operating_cost + network_violation

        soc_violation, terminal_soc_deviation, soc_min, soc_max = self._soc_metrics(dispatch_kw)
        dispatch_abs = dispatch_kw.abs()
        result = {
            "objective": objective,
            "operating_cost": operating_cost,
            "energy_cost": energy_cost,
            "curtailment_cost": curtailment,
            "degradation": degradation,
            "dispatch_abs_mean": dispatch_abs.mean(dim=(1, 2)),
            "dispatch_abs_max": dispatch_abs.amax(dim=(1, 2)),
            "charge_energy_kwh": self.interval_hours * torch.relu(-dispatch_kw).sum(dim=(1, 2)),
            "discharge_energy_kwh": self.interval_hours * torch.relu(dispatch_kw).sum(dim=(1, 2)),
            "soc_min": soc_min,
            "soc_max": soc_max,
            "network_violation": network_violation,
            "voltage_violation": voltage_violation,
            "line_flow_violation": line_flow_violation,
            "voltage_risk": voltage_violation,
            "thermal_risk": line_flow_violation,
            "kw_violation": torch.relu(dispatch_kw.abs() - self.kw_rated[None, None, :]).mean(dim=(1, 2)),
            "soc_violation": soc_violation,
            "terminal_soc_deviation": terminal_soc_deviation,
        }
        return result

    def _expand_load(self, load_forecast: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_kw = load_forecast[..., 0].sum(dim=2)
        base_total = self.base_kw.sum().clamp_min(1e-6)
        scale = total_kw / base_total
        return scale[..., None] * self.base_kw, scale[..., None] * self.base_kvar

    def _energy_price(self, horizon: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.energy_price_profile_per_kwh is None:
            return torch.full((horizon,), self.energy_price_per_kwh, device=device, dtype=dtype)

        profile = torch.tensor(self.energy_price_profile_per_kwh, device=device, dtype=dtype)
        if profile.numel() == horizon:
            return profile
        if profile.numel() == 24 and horizon % 24 == 0:
            return profile.repeat_interleave(horizon // 24)
        if profile.numel() == 1:
            return profile.expand(horizon)
        raise ValueError(
            "energy_price_profile_per_kwh must have length 1, 24, or match horizon_steps. "
            f"Got length {profile.numel()} for horizon {horizon}."
        )

    def _soc_metrics(self, dispatch_kw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        discharge_coeff = self.interval_hours / (self.kwh_rated[None, None, :] * self.discharge_efficiency[None, None, :])
        charge_coeff = self.interval_hours * self.charge_efficiency[None, None, :] / self.kwh_rated[None, None, :]
        discharge_kw = torch.relu(dispatch_kw)
        charge_kw = torch.relu(-dispatch_kw)
        delta = -discharge_coeff * discharge_kw + charge_coeff * charge_kw
        soc = self.soc_initial[None, None, :] + torch.cumsum(delta, dim=1)
        soc_violation = (
            torch.relu(self.soc_min[None, None, :] - soc) + torch.relu(soc - self.soc_max[None, None, :])
        ).mean(dim=(1, 2))
        terminal_soc_deviation = torch.relu(
            (soc[:, -1, :] - self.soc_initial[None, :]).abs() - self.terminal_soc_tolerance
        ).mean(dim=1)
        return soc_violation, terminal_soc_deviation, soc.amin(dim=(1, 2)), soc.amax(dim=(1, 2))


def _as_battery_tuple(battery: BatteryConfig | tuple[BatteryConfig, ...]) -> tuple[BatteryConfig, ...]:
    if isinstance(battery, BatteryConfig):
        return (battery,)
    return tuple(battery)


def _topological_edge_order(parent_idx: list[int], child_idx: list[int], root_idx: int) -> list[int]:
    outgoing: dict[int, list[int]] = {}
    for edge, parent in enumerate(parent_idx):
        outgoing.setdefault(parent, []).append(edge)

    order: list[int] = []
    stack = list(reversed(outgoing.get(root_idx, [])))
    while stack:
        edge = stack.pop()
        order.append(edge)
        child = child_idx[edge]
        stack.extend(reversed(outgoing.get(child, [])))

    if len(order) != len(parent_idx):
        raise ValueError("LinDistFlow network must be a connected radial tree.")
    return order
