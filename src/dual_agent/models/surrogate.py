from __future__ import annotations

import torch
from torch import nn

from dual_agent.config import BatteryConfig
from dual_agent.models.graph import GraphConv, normalize_adjacency


class OPFSurrogate(nn.Module):
    """Dispatch network used before the differentiable LinDistFlow layer.

    The surrogate maps PV/load scenarios and grid context to battery dispatch.
    Surrogate training feeds that dispatch into a Torch LinDistFlow evaluator
    and optimizes the resulting operating cost and violation penalties.
    """

    def __init__(
        self,
        pv_nodes: int,
        load_nodes: int,
        horizon_steps: int,
        scenarios: int,
        load_feature_dim: int,
        hidden_dim: int,
        graph_layers: int = 2,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        battery_count: int | None = None,
        battery: BatteryConfig | tuple[BatteryConfig, ...] | None = None,
        interval_hours: float | None = None,
    ) -> None:
        super().__init__()
        batteries = _as_battery_tuple(battery)
        if battery_count is None:
            battery_count = len(batteries) if batteries else 1
        self.pv_nodes = pv_nodes
        self.load_nodes = load_nodes
        self.horizon_steps = horizon_steps
        self.scenarios = scenarios
        self.battery_count = battery_count
        self.interval_hours = interval_hours
        self.battery_feasibility_enabled = bool(batteries) and interval_hours is not None
        if self.battery_feasibility_enabled and battery_count != len(batteries):
            raise ValueError("battery_count must match the number of configured batteries.")
        if self.battery_feasibility_enabled and interval_hours <= 0:
            raise ValueError("interval_hours must be positive when battery feasibility projection is enabled.")

        if batteries:
            self.register_buffer("kw_rated", torch.tensor([item.kw_rated for item in batteries], dtype=torch.float32))
            self.register_buffer("kwh_rated", torch.tensor([item.kwh_rated for item in batteries], dtype=torch.float32))
            self.register_buffer(
                "soc_initial", torch.tensor([item.soc_initial for item in batteries], dtype=torch.float32)
            )
            self.register_buffer("soc_min", torch.tensor([item.soc_min for item in batteries], dtype=torch.float32))
            self.register_buffer("soc_max", torch.tensor([item.soc_max for item in batteries], dtype=torch.float32))
            self.register_buffer(
                "charge_efficiency", torch.tensor([item.charge_efficiency for item in batteries], dtype=torch.float32)
            )
            self.register_buffer(
                "discharge_efficiency",
                torch.tensor([item.discharge_efficiency for item in batteries], dtype=torch.float32),
            )
        else:
            self.kw_rated = None
            self.kwh_rated = None
            self.soc_initial = None
            self.soc_min = None
            self.soc_max = None
            self.charge_efficiency = None
            self.discharge_efficiency = None
        self.projection_soc_margin = 0.0

        # Per node/time features are scenario mean, scenario std, and load features.
        input_dim = 2 + load_feature_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.graph_layers = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim, dropout=dropout) for _ in range(graph_layers)]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, min(8, hidden_dim // 16)),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)

        self.dispatch_head = nn.Linear(hidden_dim, battery_count)
        self.cost_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.risk_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))

    def forward(
        self,
        pv_scenarios: torch.Tensor,
        load_forecast: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # pv_scenarios: [batch, scenarios, horizon, nodes]
        # load_forecast: [batch, horizon, nodes, load_feature_dim]
        pv_mean = pv_scenarios.mean(dim=1)
        pv_std = pv_scenarios.std(dim=1, unbiased=False)

        if load_forecast.size(2) != pv_mean.size(-1):
            raise ValueError("For the first scaffold, load_forecast nodes must align with pv nodes.")

        x = torch.cat([pv_mean[..., None], pv_std[..., None], load_forecast], dim=-1)
        x = self.input_projection(x)

        adjacency = normalize_adjacency(adjacency).to(x.device)
        for layer in self.graph_layers:
            x = x + layer(x, adjacency)

        batch, horizon, nodes, hidden = x.shape
        pooled_nodes = x.mean(dim=2)
        encoded = self.temporal_encoder(pooled_nodes)

        raw_dispatch = self.dispatch_head(encoded)
        dispatch = self._project_battery_feasible_dispatch(raw_dispatch)
        summary = encoded.mean(dim=1)
        cost = torch.nn.functional.softplus(self.cost_head(summary)).squeeze(-1)
        risk = torch.nn.functional.softplus(self.risk_head(summary))
        return {
            "dispatch": dispatch,
            "raw_dispatch": raw_dispatch,
            "cost": cost,
            "voltage_risk": risk[:, 0],
            "thermal_risk": risk[:, 1],
        }

    def _project_battery_feasible_dispatch(self, raw_dispatch: torch.Tensor) -> torch.Tensor:
        if not self.battery_feasibility_enabled:
            return raw_dispatch
        if raw_dispatch.ndim != 3 or raw_dispatch.size(-1) != self.battery_count:
            raise ValueError("Battery feasible dispatch projection expects shape [batch, horizon, batteries].")

        assert self.interval_hours is not None
        assert self.kw_rated is not None
        assert self.kwh_rated is not None
        assert self.soc_initial is not None
        assert self.soc_min is not None
        assert self.soc_max is not None
        assert self.charge_efficiency is not None
        assert self.discharge_efficiency is not None

        soc_lower, soc_upper = self._projection_soc_bounds()
        desired_dispatch = self.kw_rated[None, None, :] * torch.tanh(raw_dispatch)
        dispatch = self._project_dispatch_kw(desired_dispatch, soc_lower, soc_upper)
        corrected_dispatch = self._apply_terminal_soc_correction(dispatch)
        return self._project_dispatch_kw(corrected_dispatch, soc_lower, soc_upper)

    def _projection_soc_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.soc_initial is not None
        assert self.soc_min is not None
        assert self.soc_max is not None

        lower = self.soc_min + self.projection_soc_margin
        upper = self.soc_max - self.projection_soc_margin
        valid_margin = (lower < self.soc_initial) & (self.soc_initial < upper)
        return torch.where(valid_margin, lower, self.soc_min), torch.where(valid_margin, upper, self.soc_max)

    def _project_dispatch_kw(
        self,
        desired_dispatch_kw: torch.Tensor,
        soc_lower: torch.Tensor,
        soc_upper: torch.Tensor,
    ) -> torch.Tensor:
        assert self.interval_hours is not None
        assert self.kw_rated is not None
        assert self.kwh_rated is not None
        assert self.soc_initial is not None
        assert self.charge_efficiency is not None
        assert self.discharge_efficiency is not None

        dispatch_steps = []
        soc = self.soc_initial[None, :].expand(desired_dispatch_kw.size(0), -1)
        kw_rated = self.kw_rated[None, :]
        for step in range(desired_dispatch_kw.size(1)):
            max_discharge_kw = torch.minimum(
                kw_rated,
                torch.relu(soc - soc_lower)
                * self.kwh_rated[None, :]
                * self.discharge_efficiency[None, :]
                / self.interval_hours,
            )
            max_charge_kw = torch.minimum(
                kw_rated,
                torch.relu(soc_upper - soc)
                * self.kwh_rated[None, :]
                / (self.charge_efficiency[None, :] * self.interval_hours),
            )
            requested_dispatch_kw = desired_dispatch_kw[:, step, :]
            dispatch_kw = torch.minimum(torch.relu(requested_dispatch_kw), max_discharge_kw) - torch.minimum(
                torch.relu(-requested_dispatch_kw),
                max_charge_kw,
            )
            discharge_kw = torch.relu(dispatch_kw)
            charge_kw = torch.relu(-dispatch_kw)
            soc = soc - (
                self.interval_hours * discharge_kw / (self.kwh_rated[None, :] * self.discharge_efficiency[None, :])
            ) + (self.interval_hours * self.charge_efficiency[None, :] * charge_kw / self.kwh_rated[None, :])
            dispatch_steps.append(dispatch_kw)
        return torch.stack(dispatch_steps, dim=1)

    def _apply_terminal_soc_correction(self, dispatch_kw: torch.Tensor) -> torch.Tensor:
        assert self.interval_hours is not None
        assert self.kwh_rated is not None
        assert self.soc_initial is not None
        assert self.charge_efficiency is not None
        assert self.discharge_efficiency is not None

        final_soc = self._final_soc(dispatch_kw)
        soc_gap = self.soc_initial - final_soc
        horizon = dispatch_kw.size(1)
        discharge_correction_kw = (
            torch.relu(-soc_gap)
            * self.kwh_rated[None, :]
            * self.discharge_efficiency[None, :]
            / (self.interval_hours * horizon)
        )
        charge_correction_kw = (
            torch.relu(soc_gap)
            * self.kwh_rated[None, :]
            / (self.charge_efficiency[None, :] * self.interval_hours * horizon)
        )
        correction_kw = discharge_correction_kw - charge_correction_kw
        return dispatch_kw + correction_kw[:, None, :]

    def _final_soc(self, dispatch_kw: torch.Tensor) -> torch.Tensor:
        assert self.interval_hours is not None
        assert self.kwh_rated is not None
        assert self.soc_initial is not None
        assert self.charge_efficiency is not None
        assert self.discharge_efficiency is not None

        discharge_kw = torch.relu(dispatch_kw)
        charge_kw = torch.relu(-dispatch_kw)
        soc_delta = -(
            self.interval_hours
            * discharge_kw
            / (self.kwh_rated[None, None, :] * self.discharge_efficiency[None, None, :])
        ) + (
            self.interval_hours
            * self.charge_efficiency[None, None, :]
            * charge_kw
            / self.kwh_rated[None, None, :]
        )
        return self.soc_initial[None, :] + soc_delta.sum(dim=1)


def _as_battery_tuple(battery: BatteryConfig | tuple[BatteryConfig, ...] | None) -> tuple[BatteryConfig, ...]:
    if battery is None:
        return ()
    if isinstance(battery, BatteryConfig):
        return (battery,)
    return tuple(battery)
