# Phase-three adaptation diagnosis

The third-stage training loop previously enabled dropout in the module being
updated and disabled it in the frozen module. Transformer attention has its own
functional dropout; disabling only `nn.Dropout` does not remove this difference.
In alternating mode, each round always updates the forecaster first and then the surrogate.

## Training changes

Phase 3 permanently disables regular, attention and recurrent dropout during
training to keep the alternating updates deterministic.

The original training partition now supplies a separate validation subset.
Every configured stage runs its complete epoch budget. The validation LinDistFlow
objective is evaluated after each epoch, and the final saved pair is the best
complete forecaster-surrogate pair observed during the schedule, including the
initial pair. The test partition is not used for automatic model selection.

## Controlled experiment

Run from the server repository root:

```bash
python scripts/experiments/phase3_iterative_adaptation_cross_evaluation/phase3_adaptation_ablation.py \
  --config configs/ieee13.yaml \
  --seeds 7 17 27 37 47 \
  --arms alternating \
  --output-dir figures/phase3_iterative_adaptation_cross_evaluation
python scripts/experiments/phase3_iterative_adaptation_cross_evaluation/aggregate_phase3_matrices.py --input-dir figures/phase3_iterative_adaptation_cross_evaluation
```

The experiment uses 300 F epochs and 200 S epochs across five rounds. Controls
use 300 F epochs (equal forecaster budget) and 500 F epochs (equal number of
optimizer updates). Epochs use the same batches per epoch; wall-clock cost is
not necessarily equal. Learning rates remain constant in this diagnostic        
experiment. Production training uses one continuous forecaster scheduler across
all phase-3 rounds.

All arms share pretrained checkpoints and train/validation/test partitions.
Seed changes affect third-stage optimization only, not independent pretraining.
Each surrogate stage is assessed with the forecaster fixed. Selection includes
the pre-update baseline; both validation objective and operating cost must
improve, without worsening voltage, flow, battery-power or SOC violations beyond
1e-6. Terminal SOC deviation remains a soft penalty and is reported separately.

`--dropout` enables the original stochastic behavior for a matched ablation.
The initial `adaptation_pilot` disabled regular/recurrent dropout but left
attention dropout enabled. `adaptation_attention_pilot` also disabled attention
dropout. Preserve both outcomes rather than selecting only successful runs.

## Interpretation limits

Validation samples are held out only during phase three. Existing pretrained
checkpoints may have seen them. The existing test partition has also been
examined during previous development. These are diagnostic results, not a new
untouched publication test. A final paper needs a fixed three-way split before
all pretraining and independently retrained seeds, or a new untouched dataset.

Lower operating cost alone does not establish physical superiority: compare
terminal SOC and all constraint metrics. Means and sample standard deviations
describe the five optimization seeds; they are not confidence bounds for unseen
operating conditions. A non-improving stage is rejected, not hidden.
