# Dual-Agent Stochastic OPF

This repository scaffolds a dual-agent system for distribution energy management:

- **Agent 1:** a spatio-temporal probabilistic PV scenario forecaster.
- **Agent 2:** a dispatch network trained through a differentiable LinDistFlow layer.

The intended workflow is:

1. Generate PV/load scenario data on the IEEE 13-bus feeder.
2. Pretrain the PV forecaster with scenario/forecast losses.
3. Train the OPF surrogate through a differentiable LinDistFlow layer using actual-PV operating cost and violation penalties.
4. Run phase-3 fine-tuning in `forecast_only`, `surrogate_only`, or `alternating` mode.

Steps 2 and 3 can be run in parallel after the dataset exists.

## Setup

```bash
pip install -e ".[dev]"
pip install -e ".[opendss]"
pip install -e ".[tracking]"
```

OpenDSS data is not vendored here. Put the IEEE 13-bus OpenDSS master file at:

```text
data/ieee13/IEEE13Nodeckt.dss
```

or pass another path through `configs/ieee13.yaml`.

## Smoke Test

```bash
pytest
```

## Generate Dataset

```bash
python scripts/data/generate_dataset.py --config configs/ieee13.yaml --out data/ieee13/ieee13.npz
```

The dataset contains `pv_history`, `pv_target`, and `load_forecast`. The main training workflow
uses the forecaster to produce scenarios, so saved dataset `pv_scenarios` are optional and are
generated only when `problem.generate_dataset_scenarios` is true. The workflow does not require
offline OPF dispatch or cost labels.

## Dataset Split

The current workflow uses a reproducible random split configured in
configs/ieee13.yaml:

    data_split:
      train_fraction: 0.64
      validation_fraction: 0.16
      test_fraction: 0.20

The same project seed controls the random partition in all training stages.

## Train Forecaster

```bash
python scripts/training/train_forecaster.py --config configs/ieee13.yaml --data data/ieee13/ieee13.npz --checkpoint checkpoints/forecaster.pt --swanlab
```

## Train Surrogate

```bash
python scripts/training/train_surrogate.py --config configs/ieee13.yaml --data data/ieee13/ieee13.npz --forecaster-checkpoint checkpoints/forecaster.pt --checkpoint checkpoints/surrogate.pt --swanlab
```

The surrogate dispatch network is trained end-to-end through a Torch LinDistFlow layer. Its loss is
`operating_cost_loss_weight * operating_cost + voltage_violation_loss_weight * voltage_violation + line_flow_violation_loss_weight * line_flow_violation + kw_violation_loss_weight * kw_violation + soc_violation_loss_weight * soc_violation + terminal_soc_loss_weight * terminal_soc_deviation`,
evaluated against `pv_target` as the actual PV trajectory. When `--forecaster-checkpoint` is
provided, the surrogate is trained from frozen forecaster-generated PV scenarios; otherwise it falls
back to the dataset `pv_scenarios`.
`operating_cost` uses the time-varying `lindistflow.energy_price_profile_per_kwh` for grid import
energy, adds battery degradation, and adds `lindistflow.curtailment_per_kwh` on reverse root-flow
PV surplus as a differentiable curtailment proxy.
`terminal_soc_deviation` is penalized only outside the `lindistflow.terminal_soc_tolerance` band
around the initial SOC, so normal daily cycling is not discouraged as strongly as true SOC-limit
violations.

## Joint Training

```bash
python scripts/training/train_joint.py --config configs/ieee13.yaml --data data/ieee13/ieee13.npz --forecaster-checkpoint checkpoints/forecaster.pt --surrogate-checkpoint checkpoints/surrogate.pt --checkpoint checkpoints/dual_agent.pt --phase3-checkpoint-dir checkpoints/phase3_states --swanlab
```

Each trainer logs train/loss, validation/loss, test/loss, and learning_rate to SwanLab when --swanlab is set.
Surrogate and joint training also log actual-PV operating cost, voltage violation, line-flow
violation, kW violation, SOC violation, terminal SOC deviation, and curtailment cost on the train, validation, and test splits where applicable. Joint training
minimizes `lindistflow_loss` with a soft forecast-loss constraint:
`lindistflow_loss + joint_forecast_constraint_weight * max(0, forecast_loss - forecast_loss_cap)`,
where `forecast_loss_cap` is the initial train forecast loss multiplied by
`1 + joint_forecast_loss_tolerance`. It logs `forecast_loss`, `forecast_loss_cap`,
`forecast_constraint_violation`, and `lindistflow_loss` to show whether the forecast constraint is
active.

Phase 3 is controlled by explicit `phase3_*` fields in `configs/ieee13.yaml`:

```yaml
training:
  phase3_mode: alternating
  phase3_forecaster_epochs: 5000
  phase3_surrogate_epochs: 5000
  phase3_rounds: 10
```

`phase3_mode: forecast_only` freezes the pretrained surrogate and fine-tunes only the forecaster
for `phase3_forecaster_epochs`. `phase3_mode: surrogate_only` freezes the forecaster and adapts only
the surrogate for `phase3_surrogate_epochs` using scenarios from the current forecaster.
`phase3_mode: alternating` divides `phase3_forecaster_epochs` and `phase3_surrogate_epochs` as evenly
as possible across `phase3_rounds`; each round first updates the forecaster, then freezes it while the
surrogate adapts to the resulting scenario distribution.

Both phase-3 stages log fresh measurements into the same `train/*` and `test/*` metric series. Use
`training_phase` to distinguish them: `0` means surrogate adaptation and `1` means forecaster
fine-tuning. Phase 3 forecaster updates use `training.phase3_forecaster_learning_rate`; phase-3 surrogate
updates use `training.phase3_surrogate_learning_rate`. The phase-3 schedule is defined only by
`phase3_mode`, `phase3_forecaster_epochs`, `phase3_surrogate_epochs`, and `phase3_rounds`;
legacy `joint_*` schedule fields are no longer part of the configuration.

Epoch counts for the three main phases are `training.forecaster_epochs`,
`training.surrogate_epochs`, and the phase-3 `training.phase3_*_epochs` fields. Phase 1 and Phase 2 use `training.forecaster_learning_rate` and `training.surrogate_learning_rate`, respectively. Every configured epoch is executed. Standalone training saves the validation-best model after the full epoch budget, while phase 3 saves the best complete forecaster-surrogate pair selected by the validation objective.

During phase 3, the validation-best model at the end of every forecaster and
surrogate stage is saved under `checkpoints/phase3_states/` as
`forecaster_00.pt` ... `forecaster_05.pt` and `surrogate_00.pt` ...
`surrogate_05.pt`. The `00` files are the initial phase-3 states. The
`checkpoints/dual_agent.pt` file remains the globally best complete pair.

## Cross-Evaluation Matrix

Evaluate all `S_i/F_j` combinations without retraining phase 3:

```bash
python scripts/experiments/phase3_iterative_adaptation_cross_evaluation/phase3_cross_evaluation.py \
  --config configs/ieee13.yaml \
  --data data/ieee13/ieee13.npz \
  --phase3-checkpoint-dir checkpoints/phase3_states \
  --output-dir figures/phase3_iterative_adaptation_cross_evaluation
```

The experiment writes only the final SVG matrix to
`figures/phase3_iterative_adaptation_cross_evaluation/phase3_cross_evaluation_matrix.svg`.
Subsequent matrix evaluations load the saved phase-3 checkpoints directly and
do not rerun phase-3 training. The number of evaluated rounds is inferred from
the contiguous checkpoint files, so no separate `--rounds` argument is needed.

## Optional Baseline Teacher

The Pyomo/HiGHS stochastic LinDistFlow teacher remains in
`src/dual_agent/opendss/lindistflow_teacher.py` as backup code for baseline experiments. It is not
used by the main training scripts. Install its solver dependencies only when needed:

```bash
pip install -e ".[baseline]"
```
