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
is used only in evaluation. The baseline is the unaltered deployed forecaster
output: it is never clipped. For each scenario/time, retain a perturbation
location only when it is daytime and ALL stations have headroom for the full
positive and negative perturbation. This common mask is fixed before any
perturbation or dispatch. Perturb one station by +/- epsilon times its capacity
on that support, without clipping. Other stations, loads and realized PV remain
fixed. Baseline predictions already outside capacity are retained, not silently
corrected; those locations are excluded from the perturbation support for every
station. This is a controlled local sensitivity experiment on common feasible
support, not an all-hours error stress test.

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
`(|C_plus-C_baseline|+|C_minus-C_baseline|)/(2*epsilon*support_fraction)`.
The support fraction is the fraction of daytime scenario/time entries retained.
It and the normalization denominator are identical for every station within a
sample. Samples with no support contribute zero (fractions are embedded in SVG).
Each vector is divided by its own maximum. The figure compares importance and
descending ranks, with Spearman correlation. Differences below 0.01% of the
maximum importance are treated as numerical ties; an entirely tied method has
no defined ranking correlation. Error bars show paired sample-bootstrap 95%
intervals, using 2,000 resamples. This is uncertainty over evaluation samples,
not a multi-training-seed experiment.

This measures sensitivity to equal relative-capacity prediction biases. It mixes
station size and network effects; ranking agreement alone does not establish
causal physical fidelity or superior cost. There are only four stations. The
surrogate is trained on actual-PV costs, while the optimizer minimizes expected
forecast-scenario cost; equal objective weights do not make these two learning
problems identical. Absolute sensitivities need not agree even if ranks agree.

## Root-cause audit

The original experiment is preserved in Git commit `f991c5c`; it clipped both
the baseline and each signed perturbation independently. On the same first 32
validation samples, the old protocol gave Spearman 0.00 and the corrected
protocol gave 1.00. This is an experiment-protocol correction, NOT a model or
training improvement; the two results use different perturbation definitions.

Across all validation samples, baseline clipping changed surrogate dispatch by
8.72 kW on average and changed mean actual operating cost from 1750.78 to
1765.26. It affected 2.52% of forecast elements. The old full-test perturbations
were clipped in 11.87% of active entries, with station-dependent asymmetry.

The deployed forecaster emits identical scenarios at all four stations
(maximum station spread exactly zero). Its complete-graph averaging removes
node distinctions. The surrogate also uses complete-graph message passing and
node-mean pooling. The current lossless LinDistFlow operating cost depends on
aggregate PV; all observed voltage and line penalties were zero. Swapping actual
station trajectories at fixed dispatch changed validation cost by only about
4e-14. These facts prevent attributing the observed ranking to network location.

An equal-kW control on the same 32 validation samples gave OPF importance
146.634709 for every station. Surrogate importance ranged only from 486.766025
to 486.774621 (relative spread 0.0018%), which is a numerical tie under the
specified rule. Relative-capacity rankings largely reflect capacity, not
electrical centrality. A high rho here must NOT be reported as proof that the
current model identifies topology-dependent critical stations.

Run the controlled validation and equal-power audits with:

```bash
python scripts/experiments/pv_station_mechanistic_consistency/station_consistency.py --split validation --samples 32 --output-dir /tmp/station_validation_corrected
python scripts/experiments/pv_station_mechanistic_consistency/station_consistency.py --split validation --samples 32 --perturbation equal-power --output-dir /tmp/station_equal_power
```

The original checkpoint is unchanged. Testing topology-dependent criticality
would require a separately defined network-sensitive study (e.g., binding
network constraints or modeled losses), and a model retaining station identity;
it cannot be established by selecting a favorable perturbation amplitude here.

Dependencies are isolated in `.venv_station` (system packages plus cvxpy/highspy).
The original project's code, checkpoints and output files are not modified.

## Corrected full-test result

With epsilon=5% and all 205 held-out samples, both methods rank
`pv_671 > pv_675 > pv_634 > pv_680` (Spearman 1.00). The mean common support is
58.41% of daytime scenario/time entries. No applied perturbations are clipped.

| Station | OPF normalized | Surrogate normalized |
| --- | ---: | ---: |
| pv_671 | 1.0000 | 1.0000 |
| pv_675 | 0.8996 | 0.9043 |
| pv_634 | 0.8076 | 0.7642 |
| pv_680 | 0.5235 | 0.4328 |

This supports relative-capacity sensitivity ranking agreement on the specified
common support. It does not prove topology-dependent station identification,
matching absolute sensitivities, or improved model accuracy. No checkpoint,
training objective, test sample membership, or epsilon was tuned to improve rho.
The protocol was corrected using validation audits before the final test rerun.
Because the earlier test result had already been inspected, a fresh independent
dataset is preferable for a confirmatory publication experiment.

## Publication figure

The SVG is formatted at 181.9 x 76.2 mm for a two-column paper. Times New Roman
and STIX mathematical glyphs are stored as vector outlines for portable font
appearance. Panel (a) retains the paired-bootstrap intervals; panel (b) has equal
rank-axis scaling. Experimental settings belong in the caption rather than a
large in-figure title. To regenerate the layout from the embedded numerical
results without rerunning model evaluation:

```bash
python scripts/experiments/pv_station_mechanistic_consistency/station_consistency.py --replot-svg figures/pv_station_mechanistic_consistency/pv_station_mechanistic_consistency.svg
```

Suggested caption:

Relative-capacity sensitivity of PV stations under forecast perturbations.
(a) Station importance, independently normalized by the maximum importance of
each method; error bars show 95% paired-bootstrap intervals (2,000 resamples).
(b) Agreement between station rankings (rank 1 denotes the highest importance).
Results use 205 test samples and positive/negative perturbations of 5% of station
capacity on common feasible daytime scenario-time entries (mean coverage 58.4%).
The comparison evaluates sensitivity under this controlled perturbation protocol
and does not isolate the effect of network location.
