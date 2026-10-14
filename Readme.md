# Future handling onset experiment

The completed five-seed LSTM/RNN experiment is preserved in [reference_results/2026-10-14](reference_results/2026-10-14). Read [EXPERIMENT_HANDOFF.md](EXPERIMENT_HANDOFF.md) for the full pasteable explanation of the changes, methodology, individual trials, class metrics, interpretations, limitations and reproduction commands. The [results summary](reference_results/2026-10-14/RESULTS.md) and [metrics CSV](reference_results/2026-10-14/metrics.csv) contain the measured outcomes.

| Model | Dry macro-F1 (mean ± SD) | Wet macro-F1 (mean ± SD) |
| --- | --- | --- |
| LSTM | 0.494 ± 0.023 | 0.522 ± 0.044 |
| Vanilla RNN | 0.503 ± 0.028 | 0.507 ± 0.030 |

To evaluate the published checkpoints without retraining, run `python testwet_lstm.py --results reference_results/2026-10-14`. New default runs write to `results_future/`; the reference snapshot remains separate.

Research question: To what extent can an LSTM trained on dry-condition telemetry predict whether physics-defined understeer or oversteer will begin within the next 0.25 seconds when tested on unseen wet-condition telemetry?

The raw semicolon-separated CSV files and existing physics labels are preserved. The old `handling_lstm.pth` belongs to current-state classification and is never loaded by this experiment.

## Reproduce

Run from this folder with Python and the packages in requirements.txt:

```
python train_lstm.py --prepare-only
python diagnose_samples.py
python audit_pipeline.py
python train_lstm.py
python report_results.py
python testwet_lstm.py
```

The default experiment runs LSTM and vanilla tanh RNN, each with seeds 42-46, 30 epochs, Adam 0.001, batch size 32, two layers, hidden width 64 and dropout 0.3. Checkpoints are selected by dry validation weighted loss. All checkpoints are frozen before wet data is loaded for evaluation. Results appear in results_future. For a smoke run use `--epochs 1 --seeds 42 --out results_smoke`; smoke results are not final evidence.

## Data and targets

Dry laps: train 1,2,3,4,6; validation 8; test 9,11. Wet laps: all nine, only final testing. Split by whole laps before constructing stride-one overlapping windows. Every input contains 30 consecutive Neutral frames from one lap. Only the original 12 features enter the model; Label, YawGradient, time, position and environmental variables do not.

CurrentTime is treated as milliseconds, consistent with approximately 86-91 second lap durations. Logging is irregular: median intervals are 18 ms, with occasional duplicates/gaps. Therefore 15 frames are not reliably 0.25 seconds. The primary protocol uses the actual interval `(input_end, input_end + 250 ms]`, with variable future frame count. Historical inputs remain 30 frames, typically about 0.48 seconds from first to last sample. Reject any candidate spanning a nonpositive interval or a gap greater than 50 ms, including the observation establishing full horizon coverage. No interpolation or row dropping silently bridges missing data.

The target is the first transition to Understeer or Oversteer followed by at least three consecutive observations of that class, all within the future horizon. If no such confirmed onset occurs, target is Neutral. A late onset whose confirmation extends beyond the horizon is not confirmed and is Neutral under this conservative operational definition. Three frames specify sample persistence, not a fixed duration. This convention must be stated in the essay. Sequential labels cannot be simultaneous; if both classes have confirmed runs, the earliest wins. This differs deliberately from assuming exactly 15 future frames.

Normalization mean/std is fitted on dry training windows only (overlapping frames contribute repeatedly). Class weights are n/(3 * training class count). Identical windows, scaler, weights and splits are used for both models. Same width/depth does not mean equal parameter count; counts are included in metrics. This baseline tests the architecture at fixed width, not a strictly parameter-matched comparison.

## Artifacts and interpretation

manifest.json records lap IDs, sample counts, features, weights, timing rules, package versions, CSV SHA-256 hashes and settings. scaler.npz stores dry-training statistics. Each checkpoint includes model type, seed, selected epoch and parameter count. Per-seed history, sample metadata, predictions, metrics.csv/json and aggregate.csv are saved. Confusion matrices in metrics.json use actual rows and predicted columns in Neutral/Understeer/Oversteer order. aggregate.csv reports mean and sample standard deviation across seeds for accuracy, macro-F1 and class precision/recall/F1 (undefined precision/recall use zero). diagnose_samples.py runs synthetic boundary tests and produces sample_audit.png.

Interpret macro-F1 and event recall alongside accuracy and the always-Neutral baseline. Evaluate the paired dry-to-wet metric change per seed. Seed variation measures training variability, not independent lap uncertainty: overlapping windows are highly dependent. Only two dry held-out laps and sparse Oversteer targets constrain conclusions; do not treat thousands of windows as independent driving trials. No wet tuning is permissible after observing final test results.

A high score alone cannot establish that ordered temporal precursors caused the result. A last-frame classifier and a temporal-order ablation would strengthen that claim. The current experiment compares LSTM/RNN temporal inputs and removes same-frame target leakage, but does not prove a memory advantage or causality. Physics label validity, threshold choices, one car/track and simulated wet conditions remain limitations. Existing CSV labels are accepted, not revalidated against their unavailable logger implementation. Prior current-state accuracy results answer a different question and must not be reported as future-onset results.
