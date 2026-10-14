# Detailed revision report: onset-v2 experiment

The original completed experiment in `reference_results/2026-10-14/` was preserved byte-for-byte. This report covers the prescribed correction and separate optional controls. New working outputs are in `results_onset_v2/`.

## Recommendations and scope decisions

| Recommendation | Decision |
| --- | --- |
| Fix onset/confirmation boundary mismatch | Required main-experiment correction; implemented and all models retrained from scratch |
| Recover original labeler | Recovered installed v3 and user-supplied GitHub v2; replayed both, documented exact rules; raw labels preserved |
| Last-frame-only baseline | Optional extension announced before implementation; fixed MLP architecture before new wet evaluation |
| Event-level evaluation | Optional secondary evaluation announced before implementation; fixed definitions recorded before training |
| Unequal recurrent parameter counts | Retained as limitation; no capacity tuning or gating-only claim |
| Wet includes grip and temperature changes | Wording/interpretation limitation; no causal claim about water alone |
| Macro-F1 and precision/recall as primary | Promoted in reports; accuracy remains supplementary; checkpoint selection still dry validation weighted loss |
| Temporal shuffling, statistical significance, parameter-matched model | Not implemented; optional follow-ups requiring a separately defined protocol |

LSTM/RNN architecture files, 12 features, lap split, seeds 42-46, Adam 0.001, batch 32, dropout 0.3 and 30 epochs were retained. No architecture/hyperparameter search was conducted. No settings were chosen or altered based on new wet performance. The wet recordings have already been used in the prior experiment, so this is a prescribed correction on reused test data rather than a newly collected pristine holdout.

## The corrected target rule

Every input remains 30 consecutive Neutral telemetry observations (~0.5 s), within one lap. A valid onset must start strictly after the input end and at most 250 ms later. The next two matching observations may extend beyond 250 ms solely to confirm persistence. An onset after 250 ms is not positive. The first confirmed onset wins; a one-frame fluctuation is not enough.

Future feature rows never enter the model input. The candidate requires full same-lap coverage of the potential confirmation tail even when its target is Neutral. Duplicate/nonpositive time increments or gaps >50 ms across input, horizon and required tail invalidate the candidate. This slightly reduces sample counts in addition to changing late-onset targets. Confirmation tails do not cross laps.

The new manifest records `confirmation_policy=onset_within_horizon`. Legacy archive evaluation explicitly selects its original `within_horizon` rule. That compatibility check reproduced all ten original wet evaluations exactly without archive writes.

## Recovered labeler and provenance

The installed logger was found under Assetto Corsa's `apps/python/LapTimeML/LapTimeML.py`. A byte-preserved copy is in `labeling_source/LapTimeML_installed_v3.py`. The user-supplied repository was inspected at https://github.com/GTs5Devy/Assetto_Corsa_Telemetry_Extraction/tree/1a76226bebad89aff838bb030d623078d9caeebf; its preserved file is `labeling_source/LapTimeML_github_v2.py`.

The installed version uses a 20-observation history and yaw-rate filter >2.0; the older GitHub version uses five observations and >1.0. The installed function matched all audited dry/wet Labels and YawGradient values. The older GitHub function did not. The first 20 observations per retained lap were excluded from the replay agreement claim because preceding context and reset ordering are not fully reconstructible after lap selection.

| Source | Domain | Audited rows | Matching labels | Matching YawGradient |
| --- | --- | --- | --- | --- |
| installed_v3 | dry | 41592 | 41592 | 41592 |
| installed_v3 | wet | 47097 | 47097 | 47097 |
| github_v2 | dry | 41592 | 39915 | 814 |
| github_v2 | wet | 47097 | 45603 | 561 |

The SlipDiff calculation also matched every raw dry/wet row. Exact thresholds, branch order and formulas are documented in PHYSICS_LABELER.md. AST extraction avoids importing/executing the live AC application. Source hashes and disagreement diagnostics are in labeler_verification.json. No thresholds were fitted and no CSV labels regenerated. This supports implementation provenance, not independent physical validity or a validated direct derivation from Bergman's theory. The filters return Neutral for some spins/decelerations, so Neutral is an operational class, not a guarantee of safe physical behavior.

## What changed at the sample level

| Split | Old windows | New windows | Dropped for extra coverage | Changed targets | Neutral→Understeer | Neutral→Oversteer |
| --- | --- | --- | --- | --- | --- | --- |
| train | 13176 | 13169 | 7 | 97 | 83 | 14 |
| validation | 2491 | 2489 | 2 | 22 | 20 | 2 |
| dry_test | 5229 | 5228 | 1 | 44 | 40 | 4 |
| wet_test | 20923 | 20915 | 8 | 179 | 145 | 34 |

Corrected class counts:

| Split | Windows | Neutral | Understeer | Oversteer |
| --- | --- | --- | --- | --- |
| train | 13169 | 12,404 (94.19%) | 619 (4.70%) | 146 (1.11%) |
| validation | 2489 | 2,326 (93.45%) | 140 (5.62%) | 23 (0.92%) |
| dry_test | 5228 | 4,897 (93.67%) | 291 (5.57%) | 40 (0.77%) |
| wet_test | 20915 | 19,567 (93.55%) | 1,061 (5.07%) | 287 (1.37%) |

Training-only weighted-cross-entropy weights in Neutral/Understeer/Oversteer order: 0.353891, 7.091546, 30.066210.


Mean/std normalization was refitted using only corrected dry training windows. All models receive that same scaler. Overlapping observations contribute repeatedly as in the original protocol. Class weights were recalculated from corrected dry training labels only. The same whole-lap split was retained: train 1,2,3,4,6; validation 8; dry test 9,11; all nine wet laps test-only.

## Optional last-frame control

The static model is Linear(12,64) → ReLU → Dropout(0.3) → Linear(64,3), with 1,027 parameters. It receives only observation 30, not the preceding 29. Its test verifies that changing those 29 observations leaves evaluation output exactly unchanged. It uses identical eligible examples, targets, split, scaler, weights, seeds, optimizer, batch size, epochs and checkpoint-selection procedure. LSTM has 53,443 parameters and RNN 13,507; model capacities are not matched.

## Fresh model results

All fifteen models were initialized from scratch and trained for 30 epochs. Dry validation weighted loss selected checkpoints; all checkpoints were frozen before new wet predictions. Every checkpoint was evaluated on both domains, producing thirty evaluation records. ± denotes sample SD across five seeds, not independent-lap confidence intervals.

Macro-F1 and class precision/recall are primary. Accuracy is supplementary.

| Model | Domain | Macro-F1 | Accuracy (%) |
| --- | --- | --- | --- |
| lstm | dry_test | 0.543 ± 0.039 | 87.96 ± 3.42 |
| lstm | wet_test | 0.553 ± 0.032 | 86.77 ± 3.09 |
| rnn | dry_test | 0.496 ± 0.031 | 83.11 ± 2.22 |
| rnn | wet_test | 0.504 ± 0.032 | 77.41 ± 4.88 |
| last_frame | dry_test | 0.507 ± 0.012 | 84.19 ± 1.35 |
| last_frame | wet_test | 0.496 ± 0.016 | 76.64 ± 2.45 |

Class precision/recall/F1:

| Model | Domain | Class | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| lstm | dry_test | Neutral | 0.975 ± 0.010 | 0.897 ± 0.044 | 0.934 ± 0.021 |
| lstm | dry_test | Understeer | 0.319 ± 0.033 | 0.633 ± 0.125 | 0.419 ± 0.030 |
| lstm | dry_test | Oversteer | 0.205 ± 0.094 | 0.550 ± 0.260 | 0.276 ± 0.083 |
| lstm | wet_test | Neutral | 0.979 ± 0.004 | 0.881 ± 0.037 | 0.927 ± 0.019 |
| lstm | wet_test | Understeer | 0.297 ± 0.039 | 0.698 ± 0.082 | 0.413 ± 0.031 |
| lstm | wet_test | Oversteer | 0.236 ± 0.055 | 0.558 ± 0.165 | 0.319 ± 0.062 |
| rnn | dry_test | Neutral | 0.985 ± 0.008 | 0.834 ± 0.031 | 0.903 ± 0.015 |
| rnn | dry_test | Understeer | 0.258 ± 0.016 | 0.831 ± 0.118 | 0.392 ± 0.018 |
| rnn | dry_test | Oversteer | 0.122 ± 0.058 | 0.465 ± 0.307 | 0.192 ± 0.099 |
| rnn | wet_test | Neutral | 0.990 ± 0.005 | 0.771 ± 0.057 | 0.866 ± 0.034 |
| rnn | wet_test | Understeer | 0.195 ± 0.028 | 0.886 ± 0.077 | 0.318 ± 0.031 |
| rnn | wet_test | Oversteer | 0.235 ± 0.042 | 0.574 ± 0.128 | 0.328 ± 0.046 |
| last_frame | dry_test | Neutral | 0.987 ± 0.003 | 0.845 ± 0.017 | 0.911 ± 0.009 |
| last_frame | dry_test | Understeer | 0.275 ± 0.013 | 0.834 ± 0.049 | 0.413 ± 0.009 |
| last_frame | dry_test | Oversteer | 0.126 ± 0.017 | 0.460 ± 0.049 | 0.198 ± 0.023 |
| last_frame | wet_test | Neutral | 0.992 ± 0.002 | 0.760 ± 0.028 | 0.861 ± 0.017 |
| last_frame | wet_test | Understeer | 0.194 ± 0.018 | 0.928 ± 0.028 | 0.320 ± 0.022 |
| last_frame | wet_test | Oversteer | 0.210 ± 0.012 | 0.583 ± 0.043 | 0.308 ± 0.017 |

Every individual trial (accuracy in percent):

| Model | Seed | Selected epoch | Dry macro-F1 | Wet macro-F1 | Dry accuracy | Wet accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| lstm | 42 | 8 | 0.528 | 0.548 | 89.31 | 88.89 |
| lstm | 43 | 16 | 0.520 | 0.568 | 89.54 | 88.47 |
| lstm | 44 | 8 | 0.506 | 0.509 | 82.19 | 82.19 |
| lstm | 45 | 21 | 0.605 | 0.597 | 90.99 | 89.33 |
| lstm | 46 | 9 | 0.556 | 0.544 | 87.76 | 84.94 |
| rnn | 42 | 3 | 0.490 | 0.487 | 81.27 | 72.84 |
| rnn | 43 | 2 | 0.544 | 0.471 | 81.58 | 72.16 |
| rnn | 44 | 5 | 0.474 | 0.487 | 81.71 | 77.57 |
| rnn | 45 | 12 | 0.505 | 0.527 | 84.98 | 82.18 |
| rnn | 46 | 13 | 0.465 | 0.548 | 86.02 | 82.30 |
| last_frame | 42 | 5 | 0.506 | 0.495 | 84.16 | 75.67 |
| last_frame | 43 | 9 | 0.527 | 0.523 | 86.27 | 80.58 |
| last_frame | 44 | 7 | 0.504 | 0.490 | 84.37 | 77.27 |
| last_frame | 45 | 6 | 0.494 | 0.482 | 82.63 | 74.26 |
| last_frame | 46 | 6 | 0.505 | 0.492 | 83.49 | 75.40 |

Always-Neutral baseline:

| Domain | Accuracy (%) | Macro-F1 |
| --- | --- | --- |
| dry_test | 93.67 | 0.322 |
| wet_test | 93.55 | 0.322 |

Paired dry-minus-wet changes:

| Model | Accuracy difference (pp) | Macro-F1 difference | Understeer recall difference (pp) | Oversteer recall difference (pp) |
| --- | --- | --- | --- | --- |
| lstm | 1.19 ± 1.11 | -0.010 ± 0.024 | -6.50 ± 6.85 | -0.82 ± 23.03 |
| rnn | 5.70 ± 3.00 | -0.008 ± 0.056 | -5.52 ± 5.27 | -10.92 ± 28.37 |
| last_frame | 7.55 ± 1.17 | 0.011 ± 0.004 | -9.40 ± 2.36 | -12.26 ± 1.18 |

Paired sequence-minus-last-frame macro-F1:

| Sequence model | Domain | Difference (mean ± SD) |
| --- | --- | --- |
| lstm | dry_test | 0.036 ± 0.048 |
| lstm | wet_test | 0.057 ± 0.035 |
| rnn | dry_test | -0.011 ± 0.025 |
| rnn | wet_test | 0.007 ± 0.044 |

A positive sequence-minus-static difference favors the sequence model; a negative difference favors this static baseline. These are paired seed differences on fixed laps. They do not isolate memory from architecture/capacity and are not significance tests.

## Optional event-level results

Each eligible event is a distinct first-confirmed onset represented by at least one positive target window. Detection requires at least one matching-class prediction before that onset within the preceding 250 ms. An event counts once despite overlapping windows. The raw confirmed-onset counts reveal how many events are outside this eligible forecast scope.

Warning episodes join consecutive same-class non-Neutral predictions within the same lap at <=50 ms end-time gaps. Neutral predictions break episodes. An episode is false if none of its windows correctly predicts the first confirmed target. This can merge early false windows with a later correct warning; the saved metrics also retain false-warning window counts, so episode rates must not be read as per-window precision.

False rates use explicitly named denominators. Per-lap rates divide by recorded test laps. Full-recorded-minute rates divide by retained lap timestamp durations. Eligible-minute rates divide by continuous stretches of eligible input-end times, summing <=50 ms gaps. All rates are conditional on the Neutral-history filter, not a simulation of continuous deployment. The protocol was written before training; no probability/warning threshold was tuned.

| Model | Domain | Class | Raw onsets | Eligible onsets | Detected count (mean ± SD) | Event recall | False episodes/lap | False episodes/eligible min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | dry_test | Understeer | 96 | 23 | 19.000 ± 1.871 | 0.826 ± 0.081 | 10.200 ± 2.110 | 14.157 ± 2.928 |
| lstm | dry_test | Oversteer | 51 | 4 | 2.800 ± 0.837 | 0.700 ± 0.209 | 3.300 ± 2.361 | 4.580 ± 3.277 |
| lstm | wet_test | Understeer | 471 | 90 | 72.400 ± 5.505 | 0.804 ± 0.061 | 12.222 ± 1.597 | 18.787 ± 2.454 |
| lstm | wet_test | Oversteer | 268 | 33 | 23.000 ± 4.796 | 0.697 ± 0.145 | 5.022 ± 2.441 | 7.720 ± 3.752 |
| rnn | dry_test | Understeer | 96 | 23 | 22.400 ± 0.894 | 0.974 ± 0.039 | 18.900 ± 7.478 | 26.233 ± 10.380 |
| rnn | dry_test | Oversteer | 51 | 4 | 3.000 ± 0.707 | 0.750 ± 0.177 | 5.800 ± 1.204 | 8.050 ± 1.671 |
| rnn | wet_test | Understeer | 471 | 90 | 82.000 ± 2.000 | 0.911 ± 0.022 | 21.756 ± 11.044 | 33.440 ± 16.975 |
| rnn | wet_test | Oversteer | 268 | 33 | 24.400 ± 2.608 | 0.739 ± 0.079 | 7.600 ± 2.222 | 11.682 ± 3.415 |
| last_frame | dry_test | Understeer | 96 | 23 | 22.000 ± 0.000 | 0.957 ± 0.000 | 27.300 ± 3.718 | 37.892 ± 5.161 |
| last_frame | dry_test | Oversteer | 51 | 4 | 3.800 ± 0.447 | 0.950 ± 0.112 | 11.700 ± 2.564 | 16.239 ± 3.559 |
| last_frame | wet_test | Understeer | 471 | 90 | 83.800 ± 0.447 | 0.931 ± 0.005 | 27.333 ± 4.630 | 42.014 ± 7.117 |
| last_frame | wet_test | Oversteer | 268 | 33 | 26.000 ± 1.581 | 0.788 ± 0.048 | 12.156 ± 0.697 | 18.684 ± 1.071 |

Per-run integer detection counts, first-warning lead times, full-recorded-minute rates, eligible exposure and episode details are saved in event_metrics.csv, event detections CSVs and warning episodes CSVs. Means over seeds can be fractional; each individual run still reports integer X-of-Y event counts. Event aggregation does not convert the small fixed lap set into independent repetitions.

## Comparison with the preserved experiment

| Model | Domain | Old macro-F1 | Corrected macro-F1 | New minus old |
| --- | --- | --- | --- | --- |
| lstm | dry_test | 0.494 | 0.543 | +0.049 |
| lstm | wet_test | 0.522 | 0.553 | +0.031 |
| rnn | dry_test | 0.503 | 0.496 | -0.008 |
| rnn | wet_test | 0.507 | 0.504 | -0.004 |

This is a comparison between related but different target definitions and slightly different eligible sets. It is not a pure estimate of an architecture improvement. Target corrections alter class weights, training examples and selected checkpoints; old and new scores should not be substituted for one another.

## Interpretation of the new evidence

lstm: dry macro-F1 0.543, wet 0.553; wet-minus-dry +0.010. Dry accuracy 87.96%, wet 86.77%. Wet Understeer precision/recall 0.297/0.698; wet Oversteer precision/recall 0.236/0.558.

rnn: dry macro-F1 0.496, wet 0.504; wet-minus-dry +0.008. Dry accuracy 83.11%, wet 77.41%. Wet Understeer precision/recall 0.195/0.886; wet Oversteer precision/recall 0.235/0.574.

last_frame: dry macro-F1 0.507, wet 0.496; wet-minus-dry -0.011. Dry accuracy 84.19%, wet 76.64%. Wet Understeer precision/recall 0.194/0.928; wet Oversteer precision/recall 0.210/0.583.



The corrected LSTM has higher mean macro-F1 than both controls on both domains: 0.543 dry and 0.553 wet, compared with RNN 0.496/0.504 and last-frame MLP 0.507/0.496. Its paired macro-F1 advantage over the last-frame control is 0.036 ± 0.048 on dry and 0.057 ± 0.035 on wet. It exceeds the static control in all five wet seeds but not every dry seed. This is evidence favoring the chosen LSTM in this fixed protocol; it does not establish a capacity-controlled or statistically significant gating advantage.

The RNN does not show a consistent advantage over the static control. This weakens a general claim that any recurrent sequence model necessarily adds value. The static model retains substantial forecasting performance, so the final observation already contains predictive information.

The event summary reveals a detection/false-warning trade-off. On wet data, the LSTM detects a mean 72.4 of 90 eligible Understeer onsets and 23.0 of 33 eligible Oversteer onsets, versus the static control's 83.8 and 26.0. The LSTM nevertheless produces fewer false episodes: 12.22 Understeer and 5.02 Oversteer false episodes per lap, versus 27.33 and 12.16 for the static control. These remain frequent warnings, not deployment-ready reliability.

The eligibility restriction is substantial: dry raw confirmed counts are 96 Understeer/51 Oversteer versus 23/4 represented eligible events; wet counts are 471/268 versus 90/33. Events without eligible Neutral histories are outside the forecast protocol. High eligible-event recall must not be presented as detecting that fraction of all recorded events.

Interpret domain changes separately by metric rather than forcing a uniform wet degradation narrative. Event precision must accompany recall: detecting many events with frequent false predictions does not establish a reliable warning system. Any recurrent advantage must be assessed relative to the fixed last-frame control and its unequal capacity. Even an advantage over that control would not prove gated memory causes transfer performance; similar or better static performance would weaken the claim that the full history adds value for these chosen models. No physical explanation for a wet improvement or deterioration was tested.

## Verification completed

The existing synthetic sanity suite and the expanded unittest suite cover late confirmation, an onset exactly at 250 ms, an onset after the horizon, future-input separation, gap/lap truncation, first-confirmed-onset priority, static-history invariance, archive output rejection, event deduplication and warning-episode counting. The updated saved-data audit checks both dry and wet metadata, training-only scaler/weights, split membership, continuity through confirmation tails, onset/confirmation boundaries, data/checkpoint hashes, target/prediction alignment and trial completeness. Diagnostics include the corrected late-onset case and were visually inspected.

The existing check_csv.py was run. Frozen checkpoint reevaluation was checked against all thirty saved evaluations. Original legacy checkpoint compatibility was verified separately. The original archive's 56 files were rehashed against a pre-edit inventory; raw CSVs remained unchanged. Compilation and Git whitespace checks were run. Exact test/verification details are recorded in verification_summary.json and archive_integrity.json in the new results directory.

## Limitations and unperformed follow-ups

- Physics-rule replay verifies implementation consistency, not external physical ground truth or threshold sensitivity. Warmup/boundary context is excluded from the full-agreement claim.
- Labels themselves use 20-observation temporal history and heuristic thresholds. A model may predict this operational labeling process; that is not an independent physical validation of handling onset.
- Sparse independent events and heavily overlapping windows remain, especially the four dry-test eligible Oversteer onsets.
- Forecast eligibility is determined with existing Neutral physics labels; a live system would need an additional eligibility mechanism.
- Event summaries cover represented first-confirmed onsets and a defined warning clustering rule. Unforecastable/raw events and exposure restrictions must remain visible.
- Last-frame MLP, RNN and LSTM capacities differ. No pure memory-only or gating-only causal conclusion follows.
- Dry/wet grip and temperatures change together, with possible driver/lap differences. The experiment measures recorded dry-to-wet-domain transfer, not the isolated effect of water.
- Timestamp interpretation, variable history duration, three-observation persistence and the 50 ms continuity threshold constrain the target/sample definition.
- No temporal shuffling ablation, parameter-matched search, event-level significance test, independent-lap bootstrap, probability calibration or continuous deployment simulation was performed.
- Seed SD measures optimization variability on this fixed split, not uncertainty across independent driving sessions.
- Wet data has been observed previously. This is a specified correction without new wet-based tuning, not a fresh external validation study.

## File changes

| File | Scope of revision |
| --- | --- |
| dataset_lstm.py | Target boundary correction, explicit confirmation policy/tail metadata, legacy rule support; existing features/splits preserved |
| train_lstm.py | Same core training loop; added static model option, new output default, archive/overwrite guard, protocol/source/checkpoint provenance and pre-wet frozen status |
| experiment_config.py | Shared model factory, versioned defaults, manifest-driven window policy and immutable archive output guard |
| model_last_frame.py | Optional static control using only the last past observation |
| event_evaluation.py | Predeclared event detection and warning-rate evaluation of frozen predictions |
| verify_labeler.py | AST replay and comparison of recovered local/GitHub labelers |
| labeling_source/ | Byte-preserved recovered source versions and source repository license |
| .gitattributes | Preserve recovered source bytes/line endings and their original whitespace in Git |
| audit_pipeline.py | Existing audit extended to both domains, corrected/legacy rules and complete artifact checks |
| diagnose_samples.py | Existing tests updated for corrected boundary; fourth plot panel for post-horizon confirmation; output path argument |
| test_pipeline.py | Focused regression tests for changed behavior and optional controls |
| testwet_lstm.py | Manifest-driven corrected/legacy labels and all three model types |
| report_results.py | Primary macro-F1 presentation, three models, static contrasts, event summaries and output path argument |
| PHYSICS_LABELER.md | Exact recovered rules, version difference, provenance and limitations |
| PROTOCOL_V2.md | Predeclared required change and separately identified optional controls |
| EXPERIMENT_METHOD.md, Readme.md | Updated method and instructions |
| EXPERIMENT_HANDOFF.md | Earlier handoff explicitly marked historical; original content retained |
| REVISION_REPORT.md | This detailed current report |
| model_lstm.py, model_rnn.py, raw CSVs | Unchanged |
| reference_results/2026-10-14/ | Untouched original archive |

## Reproduce and inspect

Run from the repository root with the existing requirements.txt environment. The default working output is results_onset_v2. Training rejects an existing checkpoint/metrics directory rather than overwriting evidence. Use a fresh directory name for another complete run.

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

For existing working checkpoints, reevaluate with `python testwet_lstm.py --results results_onset_v2`. For the original untouched archive, use `python testwet_lstm.py --results reference_results/2026-10-14`; the evaluator uses its original rule. Writing/generation commands reject reference_results paths. Do not tune based on already observed wet metrics.

Every metric, individual trial, confusion matrix, prediction, sample boundary, checkpoint, event detection and warning episode is available in the new results directory. The raw dataset format and labels are preserved.
