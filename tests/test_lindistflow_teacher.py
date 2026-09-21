from __future__ import annotations

import numpy as np
import pytest

from dual_agent.config import BatteryConfig, LinDistFlowConfig
from dual_agent.opendss.lindistflow_teacher import LinDistFlowStochasticTeacher, parse_ieee13_lindistflow
from dual_agent.opendss.lindistflow_torch import TorchLinDistFlowEvaluator


def test_parse_ieee13_lindistflow_network() -> None:
    network = parse_ieee13_lindistflow("data/ieee13/IEEE13Nodeckt.dss")

    assert network.root_bus == "RG60"
    assert "675" in network.buses
    assert any(branch.parent == "692" and branch.child == "675" for branch in network.branches)
    assert network.base_load_kw["675"] > 0.0


def test_lindistflow_teacher_solves_short_horizon() -> None:
    pytest.importorskip("pyomo.environ")
    teacher = LinDistFlowStochasticTeacher(
        "data/ieee13/IEEE13Nodeckt.dss",
        ("pv_634", "pv_671", "pv_675", "pv_680"),
        BatteryConfig(
            name="bat_675",
            bus="675.1.2.3",
            phases=3,
            kv=4.16,
            kw_rated=500.0,
            kwh_rated=1000.0,
            soc_initial=0.5,
            soc_min=0.1,
            soc_max=0.9,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
        ),
        LinDistFlowConfig(
            voltage_lower=0.95,
            voltage_upper=1.05,
            energy_price_per_kwh=0.12,
            degradation_per_kwh=0.02,
            curtailment_per_kwh=0.04,
        ),
        interval_hours=0.25,
    )
    pv_scenarios = np.zeros((1, 2, 4), dtype=np.float32)
    load_forecast = np.full((2, 4, 2), 100.0, dtype=np.float32)

    label = teacher.solve(pv_scenarios, load_forecast)

    assert label.dispatch_kw.shape == (2,)
    assert np.isfinite(label.expected_cost)
    assert np.all(label.dispatch_kw <= 500.0)
    assert np.all(label.dispatch_kw >= -500.0)


def test_torch_lindistflow_evaluator_is_differentiable() -> None:
    import torch

    batteries = (
        BatteryConfig(
        name="bat_675",
        bus="675.1.2.3",
        phases=3,
        kv=4.16,
        kw_rated=500.0,
        kwh_rated=1000.0,
        soc_initial=0.5,
        soc_min=0.1,
        soc_max=0.9,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        ),
        BatteryConfig(
            name="bat_680",
            bus="680.1.2.3",
            phases=3,
            kv=4.16,
            kw_rated=500.0,
            kwh_rated=1000.0,
            soc_initial=0.5,
            soc_min=0.1,
            soc_max=0.9,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
        ),
    )
    teacher = LinDistFlowConfig(
        voltage_lower=0.95,
        voltage_upper=1.05,
        energy_price_per_kwh=0.12,
        degradation_per_kwh=0.02,
        curtailment_per_kwh=0.04,
    )
    evaluator = TorchLinDistFlowEvaluator(
        "data/ieee13/IEEE13Nodeckt.dss",
        ("pv_634", "pv_671", "pv_675", "pv_680"),
        batteries,
        teacher,
        interval_hours=0.25,
    )
    dispatch = torch.zeros(2, 4, 2, requires_grad=True)
    pv_scenarios = torch.full((2, 3, 4, 4), 50.0)
    load_forecast = torch.full((2, 4, 4, 2), 100.0)

    result = evaluator(dispatch, pv_scenarios, load_forecast)
    result["objective"].mean().backward()

    assert dispatch.grad is not None
    assert torch.isfinite(dispatch.grad).all()
