# PV Station Mechanistic Consistency

Run from the copied project root `dual_agent_station_importance`:

```bash
source .venv_station/bin/activate
python scripts/experiments/pv_station_mechanistic_consistency/station_consistency.py
```

The default evaluates all held-out test samples using `checkpoints/dual_agent.pt`
and an epsilon of 0.05. Use `--samples 2` for a smoke test or `--epsilon 0.03`
for another perturbation amplitude. No models are retrained.

Only `figures/pv_station_mechanistic_consistency/pv_station_mechanistic_consistency.svg`
is written. The SVG description embeds exact scores, per-sample sensitivities,
configuration, checkpoint hash and sample indices for reproducibility.

## Protocol

Capacity comes from the synthetic dataset's historical feature 4 (MW converted
to kW), not the nameplate values of unrelated DSS PV elements. Known historical
clear-sky feature 1 at the matching time of day defines daytime; actual test PV
is used only in evaluation. The forecast is clipped to [0, capacity] once for
both baseline methods. Perturb one station by +/- epsilon times its capacity,
identically across all scenarios during daytime, then clip to physical bounds.
Report the fraction of perturbations shortened by clipping. Other inputs and
the realized PV trajectory remain fixed.

Both methods receive exactly the same scenarios. A multi-battery HiGHS MILP
uses the same feeder topology, load expansion, voltage equations, price profile,
surplus/curtailment proxy, battery efficiencies and objective weights as the
Torch evaluator. Battery power and SOC bounds are hard constraints and binary
modes prevent simultaneous charging/discharging. Voltage, line flow and terminal
SOC deviation remain soft penalties, matching the project's objective. The
terminal SOC is NOT constrained to return exactly to its initial value.
Every optimum is checked against the Torch objective and battery feasibility.
Nonoptimal solver status aborts rather than silently contributing a result.

The two dispatches are evaluated on actual PV using the same Torch LinDistFlow
cost. For each method separately, station importance is the mean of
`(|C_plus-C_baseline|+|C_minus-C_baseline|)/(2*epsilon)`.
Each vector is divided by its own maximum. The figure compares importance and
descending ranks (average ranks for ties), with Spearman correlation.

This measures sensitivity to equal relative-capacity prediction biases. It mixes
station size and network effects; ranking agreement alone does not establish
causal physical fidelity or superior cost. There are only four stations. The
baseline clip is an explicit preprocessing choice. Inspect several epsilon
values and feasibility metrics before drawing publication-level conclusions.

Dependencies are isolated in `.venv_station` (system packages plus cvxpy/highspy).
The original project's code, checkpoints and output files are not modified.
