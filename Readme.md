# Future handling onset prediction: corrected onset-v2 protocol

Research question: To what extent can an LSTM trained on dry-condition telemetry predict whether physics-defined understeer or oversteer will begin within the next 0.25 seconds when tested on unseen wet-condition telemetry?

The current protocol fixes the late-onset boundary: onset must start within 250 ms; its next two matching observations may extend after the horizon solely for persistence confirmation. Inputs contain 30 consecutive Neutral telemetry observations (~0.5 s), with the same 12 features and whole-lap splits as before. Macro-F1 and class precision/recall are primary; accuracy is supplementary.

Completed corrected results are preserved in [reference_results/2026-10-14-onset-v2](reference_results/2026-10-14-onset-v2), including all fifteen checkpoints and thirty domain evaluations. Mean ± sample SD across five seeds:

| Model | Dry macro-F1 | Wet macro-F1 |
| --- | --- | --- |
| LSTM | 0.543 ± 0.039 | 0.553 ± 0.032 |
| Vanilla RNN | 0.496 ± 0.031 | 0.504 ± 0.032 |
| Last-frame MLP | 0.507 ± 0.012 | 0.496 ± 0.016 |

Reload these checkpoints with `python testwet_lstm.py --results reference_results/2026-10-14-onset-v2`. The LSTM has higher mean macro-F1 and fewer false-warning episodes in this protocol; the static control detects more eligible events. Capacities differ, precision remains limited, and event recall covers represented eligible onsets rather than all raw events.

Read [PROTOCOL_V2.md](PROTOCOL_V2.md) for the predeclared settings and separately identified optional last-frame/event controls, [PHYSICS_LABELER.md](PHYSICS_LABELER.md) for recovered labeler rules and replay evidence, and [EXPERIMENT_METHOD.md](EXPERIMENT_METHOD.md) for variables, controls and limitations. The new detailed results report is [REVISION_REPORT.md](REVISION_REPORT.md).

## Preserved original experiment

[reference_results/2026-10-14](reference_results/2026-10-14) is the untouched onset-v1 archive. Its earlier within-horizon confirmation rule and metrics remain historical evidence. [EXPERIMENT_HANDOFF.md](EXPERIMENT_HANDOFF.md) is explicitly marked as that earlier handoff. New results use a different target definition and must not be substituted for the old scores.

The original raw CSVs and physics labels remain unchanged. The historical handling_lstm.pth is not loaded for new training.

## Reproduce the corrected protocol

Use the packages in requirements.txt. Run from the repository root. Use a new output directory for a new full run:

```powershell
python -m unittest -v test_pipeline
python verify_labeler.py --results results_onset_v2_reproduction
python train_lstm.py --prepare-only --out results_onset_v2_reproduction
python diagnose_samples.py --results results_onset_v2_reproduction
python audit_pipeline.py --results results_onset_v2_reproduction
python train_lstm.py --out results_onset_v2_reproduction
python event_evaluation.py --results results_onset_v2_reproduction
python report_results.py --results results_onset_v2_reproduction
python audit_pipeline.py --results results_onset_v2_reproduction
python testwet_lstm.py --results results_onset_v2_reproduction
```

The default working output is results_onset_v2. Training refuses to overwrite existing checkpoint/metrics outputs. Writing/generation commands reject any output under reference_results. Read-only legacy checkpoint evaluation remains supported:

```powershell
python testwet_lstm.py --results reference_results/2026-10-14
```

The manifest selects the original archive's confirmation policy so reevaluation uses the correct target definition. Existing new working checkpoints can be evaluated with `python testwet_lstm.py --results results_onset_v2`.

For a smoke test, use `python train_lstm.py --epochs 1 --seeds 42 --out results_smoke_v2`. Its results are not final evidence. All commands accepting --results use that directory explicitly; report/event commands require completed training outputs.

## Fixed data and models

Dry train laps: 1,2,3,4,6. Dry validation: 8. Dry test: 9,11. All nine wet laps are target-domain test data. Wet data never fits normalization, class weights or model parameters and does not select checkpoints/settings.

Features: YawRate, LateralAccel, LongitudinalAccel, SteerAngle, Speed, WheelSlipFL, WheelSlipFR, WheelSlipRL, WheelSlipRR, Throttle, Brake, SlipDiff. Label, YawGradient, time, position and environmental variables are excluded from model inputs.

LSTM and RNN remain two layers, hidden width 64, dropout 0.3, three outputs. The optional last-frame MLP uses only the last observation: 12->64 ReLU, dropout 0.3, 64->3. All use the same corrected samples, training-only scaler/weights, seeds 42-46, Adam 0.001, batch 32, 30 epochs, and lowest dry validation weighted-loss checkpoint. Models are trained from scratch. All checkpoints are frozen before wet-model evaluation.

Sampling is irregular, so the future interval uses timestamps rather than assuming 15 observations equal 250 ms. Candidates require sufficient same-lap coverage for the full possible confirmation tail and reject nonpositive increments or gaps >50 ms. Normalization uses only generated dry training windows, with overlap contributing repeatedly. Weights use n/(3*training class count).

## Evidence and interpretation

The run manifest records protocol, features, splits, counts, class weights, source/data hashes and frozen checkpoint hashes. Outputs include scaler, checkpoints, epoch histories, metadata, target/prediction CSVs, all class metrics and confusion matrices, aggregate mean/sample SD, paired domain differences, the Neutral baseline, static-control contrasts, event detections, warning episodes and diagnostic plots.

The installed v3 labeler reproduces all audited interior dry/wet Label and YawGradient values. The supplied GitHub source is an older v2 variant and does not fully match. Both copies and exact thresholds are preserved in labeling_source. This verifies the implemented rules, not independent physical validity.

Window metrics and optional event summaries remain conditional on known Neutral-history eligibility. Event rates use predeclared warning clustering and clearly labeled exposure denominators. Raw confirmed-onset counts are reported alongside forecastable onsets. Overlapping windows, few independent events, unequal model capacities, and grip/temperature/driver confounding constrain conclusions. Seed SD is optimization variability, not independent-lap uncertainty. No causal gating-only or rain-only claim is justified, and no new wet-based tuning is performed.
