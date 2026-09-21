from __future__ import annotations

import torch

from dual_agent.config import BatteryConfig
from dual_agent.models.dual_agent import DualAgentSystem
from dual_agent.models.forecaster import PVScenarioForecaster
from dual_agent.models.surrogate import OPFSurrogate


def test_dual_agent_forward_and_backward() -> None:
    batch = 3
    history = 6
    horizon = 8
    nodes = 4
    scenarios = 5
    pv_features = 6
    load_features = 2

    forecaster = PVScenarioForecaster(
        pv_feature_dim=pv_features,
        hidden_dim=32,
        horizon_steps=horizon,
        scenarios=scenarios,
        graph_layers=1,
        temporal_layers=1,
    )
    surrogate = OPFSurrogate(
        pv_nodes=nodes,
        load_nodes=nodes,
        horizon_steps=horizon,
        scenarios=scenarios,
        load_feature_dim=load_features,
        hidden_dim=32,
        graph_layers=1,
        temporal_layers=1,
    )
    system = DualAgentSystem(forecaster, surrogate)

    pv_history = torch.rand(batch, history, nodes, pv_features)
    load_forecast = torch.rand(batch, horizon, nodes, load_features)
    adjacency = torch.ones(nodes, nodes) - torch.eye(nodes)

    output = system(pv_history, load_forecast, adjacency)
    loss = output["cost"].mean() + output["voltage_risk"].mean()
    loss.backward()

    assert output["pv_scenarios"].shape == (batch, scenarios, horizon, nodes)
    assert output["dispatch"].shape == (batch, horizon, 1)
    assert any(param.grad is not None for param in forecaster.parameters())


def test_surrogate_battery_projection_keeps_soc_feasible() -> None:
    horizon = 8
    battery_675 = BatteryConfig(
        name="bat",
        bus="675",
        phases=3,
        kv=4.16,
        kw_rated=500.0,
        kwh_rated=1000.0,
        soc_initial=0.5,
        soc_min=0.1,
        soc_max=0.9,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )
    battery_680 = BatteryConfig(
        name="bat_680",
        bus="680",
        phases=3,
        kv=4.16,
        kw_rated=250.0,
        kwh_rated=500.0,
        soc_initial=0.45,
        soc_min=0.1,
        soc_max=0.9,
        charge_efficiency=0.9,
        discharge_efficiency=0.9,
    )
    batteries = (battery_675, battery_680)
    surrogate = OPFSurrogate(
        pv_nodes=4,
        load_nodes=4,
        horizon_steps=horizon,
        scenarios=5,
        load_feature_dim=2,
        hidden_dim=32,
        graph_layers=1,
        temporal_layers=1,
        battery=batteries,
        interval_hours=0.25,
    )
    raw_dispatch = torch.full((2, horizon, 2), 100.0)

    dispatch = surrogate._project_battery_feasible_dispatch(raw_dispatch)
    kwh_rated = torch.tensor([item.kwh_rated for item in batteries])
    soc_initial = torch.tensor([item.soc_initial for item in batteries])
    soc_min = torch.tensor([item.soc_min for item in batteries])
    soc_max = torch.tensor([item.soc_max for item in batteries])
    kw_rated = torch.tensor([item.kw_rated for item in batteries])
    charge_efficiency = torch.tensor([item.charge_efficiency for item in batteries])
    discharge_efficiency = torch.tensor([item.discharge_efficiency for item in batteries])
    discharge_coeff = 0.25 / (kwh_rated[None, :] * discharge_efficiency[None, :])
    charge_coeff = 0.25 * charge_efficiency[None, :] / kwh_rated[None, :]
    soc_delta = -discharge_coeff * torch.relu(dispatch) + charge_coeff * torch.relu(-dispatch)
    soc = soc_initial[None, None, :] + torch.cumsum(soc_delta, dim=1)

    assert torch.all(dispatch <= kw_rated[None, None, :])
    assert torch.all(dispatch >= -kw_rated[None, None, :])
    assert torch.all(soc >= soc_min[None, None, :] - 1e-6)
    assert torch.all(soc <= soc_max[None, None, :] + 1e-6)
    assert torch.allclose(soc[:, -1, :], soc_initial[None, :].expand(2, -1), atol=0.05)
