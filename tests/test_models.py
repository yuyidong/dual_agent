from __future__ import annotations

import torch

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

