from __future__ import annotations

from pathlib import Path

from dual_agent.config import load_config


def test_load_config_resolves_dss_path_from_cwd() -> None:
    config = load_config("configs/ieee13.yaml")
    assert config.problem.horizon_steps == 24
    assert config.network.dss_master == (Path.cwd() / "data/ieee13/IEEE13Nodeckt.dss").resolve()

