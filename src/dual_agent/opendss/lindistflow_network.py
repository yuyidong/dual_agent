from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
