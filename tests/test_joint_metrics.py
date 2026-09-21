from __future__ import annotations

import torch

from dual_agent.training.loops import _accumulate_joint_metrics


def test_forecast_constraint_violation_is_averaged_per_batch() -> None:
    totals: dict[str, float] = {}
    cap = 100.0

    for forecast_loss_value in (80.0, 120.0):
        forecast_loss = torch.tensor(forecast_loss_value)
        forecast_violation = torch.relu(forecast_loss - cap)
        lindistflow_loss = torch.tensor(10.0)
        loss = lindistflow_loss + forecast_violation
        _accumulate_joint_metrics(
            totals,
            result={},
            loss=loss,
            forecast_loss=forecast_loss,
            lindistflow_loss=lindistflow_loss,
            forecast_constraint_violation=forecast_violation,
            forecast_loss_cap=cap,
            batch_size=1,
        )

    samples = 2
    metrics = {key: value / samples for key, value in totals.items()}

    assert metrics["forecast_loss"] == 100.0
    assert metrics["forecast_loss_cap"] == cap
    assert metrics["forecast_constraint_violation"] == 10.0
    assert metrics["loss"] == 20.0
