# Dual-Agent Stochastic OPF

This repository scaffolds a dual-agent system for distribution energy management:

- **Agent 1:** a spatio-temporal probabilistic PV scenario forecaster.
- **Agent 2:** a differentiable neural surrogate of a multi-step stochastic OPF teacher.

The intended workflow is:

1. Generate OPF teacher labels on the IEEE 13-bus feeder with OpenDSS.
2. Pretrain the PV forecaster with scenario/forecast losses.
3. Train the OPF surrogate to predict dispatch, cost, and constraint risk from PV scenarios.
4. Jointly tune the forecaster through the frozen or slowly updated OPF surrogate.

Steps 2 and 3 can be run in parallel after the dataset exists. In the closed-loop version, pretrain the forecaster first, use its scenarios to enrich the dataset with additional OPF teacher labels, then train or refresh the surrogate.

## Setup

```bash
pip install -e ".[dev]"
pip install -e ".[opendss]"
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
python scripts/generate_dataset.py \
  --config configs/ieee13.yaml \
  --out data/ieee13.npz
```

The teacher currently uses a compact stochastic MPC wrapper: it samples candidate battery schedules, evaluates them across PV/load scenarios with OpenDSS power flows, and saves the best schedule and risk/cost labels. You can later replace this teacher with a formal stochastic OPF solver while keeping the same saved dataset schema.

## Train Surrogate

## Pretrain Forecaster

```bash
python scripts/pretrain_forecaster.py \
  --config configs/ieee13.yaml \
  --data data/ieee13.npz \
  --checkpoint forecaster.pt
```

## Train Surrogate

```bash
python scripts/train_surrogate.py \
  --config configs/ieee13.yaml \
  --data data/ieee13.npz
```

## Joint Training

```bash
python scripts/train_joint.py \
  --config configs/ieee13.yaml \
  --data data/ieee13.npz \
  --forecaster-checkpoint forecaster.pt
```
