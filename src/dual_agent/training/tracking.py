from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class SwanLabTracker:
    def __init__(self, module: Any | None) -> None:
        self._module = module

    @property
    def enabled(self) -> bool:
        return self._module is not None

    def log(self, metrics: dict[str, float | int], step: int) -> None:
        if self._module is None:
            return
        try:
            self._module.log(metrics, step=step)
        except TypeError:
            self._module.log({"epoch": step, **metrics})

    def finish(self) -> None:
        if self._module is None:
            return
        finish = getattr(self._module, "finish", None)
        if finish is not None:
            finish()


def init_swanlab_tracker(
    enabled: bool,
    project: str,
    experiment_name: str,
    config: Any,
) -> SwanLabTracker:
    if not enabled:
        return SwanLabTracker(None)

    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError(
            "SwanLab tracking was requested, but swanlab is not installed. "
            'Install it with `pip install -e ".[tracking]"` or `pip install swanlab`.'
        ) from exc

    swanlab.init(
        project=project,
        experiment_name=experiment_name,
        config=_to_plain(config),
    )
    return SwanLabTracker(swanlab)


def _to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
