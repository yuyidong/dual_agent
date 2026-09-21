# Phase-three adaptation diagnosis

The third-stage training loop previously enabled dropout in the module being
updated and disabled it in the frozen module. Transformer attention has its own
functional dropout; disabling only `nn.Dropout` does not remove this difference.

## Training changes

`train_joint.py --phase3-disable-dropout` disables regular, attention and recurrent
dropout without freezing parameters. `--forecaster-first` runs F then S within
each alternating round. The recommended configuration enables both; command-line
boolean options can override the configuration.

The original training partition now supplies a separate validation subset.
When stage early stopping is enabled, the stage's initial model is a candidate,
and model, optimizer and scheduler states are restored together. Selection uses
the validation LinDistFlow objective, including the configured constraint terms.
The selected state is restored even when a stage exhausts its epoch budget.
The final saved pair is the best validation pair including the initial pair.
The test partition is not used for automatic model selection.

## Controlled experiment

Run from the server repository root:

```bash
python scripts/experiments/phase3_adaptation_ablation.py \
  --config configs/ieee13_phase3_recommended.yaml \
  --seeds 7 17 27 37 47 \
  --arms forecast_only forecast_budget alternating \
  --output-dir figures/adaptation_five_seeds
python scripts/experiments/summarize_adaptation.py figures/adaptation_five_seeds
```

The experiment uses 300 F epochs and 200 S epochs across five rounds. Controls
use 300 F epochs (equal forecaster budget) and 500 F epochs (equal number of
optimizer updates). Epochs use the same batches per epoch; wall-clock cost is
not necessarily equal. Learning rates remain constant in this diagnostic
experiment. Production training retains its configurable scheduler.

All arms share pretrained checkpoints and train/validation/test partitions.
Seed changes affect third-stage optimization only, not independent pretraining.
Each surrogate stage is assessed with the forecaster fixed. Selection includes
the pre-update baseline; both validation objective and operating cost must
improve, without worsening voltage, flow, battery-power or SOC violations beyond
1e-6. Terminal SOC deviation remains a soft penalty and is reported separately.
The production early-stopping rule uses the aggregate objective and is not
identical to this stricter diagnostic acceptance rule.

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
