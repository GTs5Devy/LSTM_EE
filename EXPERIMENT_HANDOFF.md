# Complete experiment handoff: future handling-onset prediction

Repository: https://github.com/GTs5Devy/LSTM_EE

Local working folder: `C:\EE\LSTM_EE_future`

Completed experiment snapshot: `reference_results/2026-10-14/`

This document records what was inspected, what changed, the exact experiment that was run, the measured results, what those results support, the important limitations, and how to reproduce the work. It is a factual handoff for the Extended Essay; it does not replace checking the physics-label definition against the original collection notes.

## 1. Research question and purpose

The revised research question is:

“To what extent can an LSTM trained on dry-condition telemetry predict whether physics-defined understeer or oversteer will begin within the next 0.25 seconds when tested on unseen wet-condition telemetry?”

The intended relationship is:

Past dry telemetry → learn patterns preceding handling onset → evaluate those patterns on held-out dry and unseen wet telemetry.

This is temporal prediction under domain shift. The environmental comparison is dry versus wet, and the architecture comparison is LSTM versus vanilla RNN. Wet data was not used to train models, fit normalization, calculate class weights, select checkpoints, or choose hyperparameters. The training program finished and froze all ten dry-selected checkpoints before loading wet telemetry for evaluation. An initial data audit inspected file contents and lap counts, including wet metadata; “unseen” here means excluded from all model fitting and selection, not that the wet file was never opened by a person or tool.

The precise implemented task is conditional forecasting: given 30 past observations that are all physics-labeled Neutral, predict the first confirmed future Understeer or Oversteer onset within a timestamp-bounded 250 ms interval. Neutral means no onset satisfying the confirmation rule.

## 2. What was wrong with the old pipeline

The original dataset assigned each 30-frame sequence the label at the final frame of that same sequence. Because inputs included variables used by the physics labeling rule, the model could learn a close approximation of the concurrent rule. This answered a current-state classification question rather than establishing advance warning.

The original code also generated windows across the whole CSV without explicitly respecting lap boundaries, randomly split overlapping sequences into training and validation, fitted dry normalization before splitting, computed class weights from the entire dry dataset, and independently normalized wet data using wet statistics. Those choices weaken split independence and the interpretation of domain transfer. There was no separate held-out dry test set in that training workflow.

The old `handling_lstm.pth` and any earlier current-state accuracy figures belong to the old task. That checkpoint remains available as a historical artifact but was not loaded for the new experiment. The old reported dry/wet results must not be presented as future-onset performance.

## 3. Data that was actually found

The raw CSVs were reused without recollecting driving data, changing their format, or replacing their frame-level labels. Both use semicolon delimiters.

| Dataset | Raw rows | Actual lap IDs | Laps |
| --- | --- | --- | --- |
| Dry | 41,752 | 1, 2, 3, 4, 6, 8, 9, 11 | 8 |
| Wet | 47,277 | 1, 4, 5, 7, 8, 9, 10, 11, 12 | 9 |

Raw frame-label counts were:

| Dataset | Neutral | Understeer | Oversteer |
| --- | --- | --- | --- |
| Dry | 31,076 | 8,281 | 2,395 |
| Wet | 32,786 | 10,386 | 4,105 |

These are current-state frame counts, not the class distribution used to train the revised model.

The recorded car and track match between conditions: `ferrari_458_gt2` at `ks_laguna_seca`. Dry road/air temperatures are 36°C/26°C; wet road/air temperatures are 20°C/20°C. Recorded SurfaceGrip is 1.0 for dry and approximately 0.9402–0.9410 for wet. Consequently, the observed environmental difference combines grip and temperature changes; the experiment does not isolate the causal effect of water alone.

The existing Label column is accepted as the physics-derived operational ground truth supplied with the dataset. The original telemetry logger/physics-label implementation is not included in the repository, so the underlying Bergman-based thresholds and their implementation were not independently verified.

## 4. Exact revised sample construction

Only these 12 features enter either neural network:

`YawRate, LateralAccel, LongitudinalAccel, SteerAngle, Speed, WheelSlipFL, WheelSlipFR, WheelSlipRL, WheelSlipRR, Throttle, Brake, SlipDiff`.

Label and YawGradient are excluded. Time, lap ID, car/track identity, track position, coordinates, heading, pitch, roll, grip and temperatures are also excluded from model inputs. CurrentTime and Lap are used only to construct and audit samples.

Every candidate input is 30 consecutive observations within one lap. All 30 original physics labels must be Neutral. If any input frame is Understeer or Oversteer, that candidate is rejected. This prevents the model from being shown an event that has already begun.

### Timing decision

The collected data is not uniformly sampled at exactly 60 Hz. Median within-lap time increments are 18 ms in both files; wet intervals vary more, and there are occasional gaps and a dry duplicate timestamp. Therefore, assuming that exactly 15 future observations always equal 0.25 seconds would not enforce the research question reliably.

CurrentTime is interpreted as milliseconds, consistent with recorded lap durations of roughly 86–91 seconds. The logger implementation is unavailable, so that unit interpretation should still be checked against the collection setup.

The implemented future interval is `(input_end_time, input_end_time + 250 ms]`. Its number of observations can vary. The input remains 30 frames: its first-to-last span is generally around half a second, but is not fixed to exactly 0.5 seconds.

Each lap is sorted separately by CurrentTime. A candidate is rejected if its input/future coverage contains a nonpositive time increment or an increment greater than 50 ms. This check also includes the next observation beyond the horizon, which establishes full coverage. Candidates near lap ends without that coverage are excluded. No interpolation is performed, and invalid rows are not silently removed and joined across a discontinuity. Missing/nonfinite required telemetry or unknown labels cause a validation error.

### Future-label decision

Starting strictly after the input, inspect future observations in chronological order. A confirmed onset is a transition to Understeer or Oversteer followed by at least three consecutive observations with that same label. All three confirming observations must lie inside the 250 ms horizon.

The target is the first such confirmed onset. If Understeer and Oversteer both have valid runs, the earliest is selected. Sequential frame labels cannot be simultaneous. If there is no confirmed onset, the target is Neutral.

Important boundary convention: an event beginning very near the horizon end is Neutral if its three-frame confirmation is not completed inside the horizon. This is a conservative operational definition and slightly narrower than “any event begins within 0.25 seconds, even if confirmation comes later.” Single-frame noise can also occur in a Neutral-target future period. Three observations define persistence in frames, not a constant number of milliseconds.

No future feature rows enter the input tensor. Future labels determine the target only.

## 5. Splits and revised class distributions

Whole laps are assigned before generating stride-one overlapping windows:

| Role | Domain | Lap IDs |
| --- | --- | --- |
| Training | Dry | 1, 2, 3, 4, 6 |
| Validation/checkpoint selection | Dry | 8 |
| Held-out same-domain test | Dry | 9, 11 |
| Unseen target-domain test | Wet | 1, 4, 5, 7, 8, 9, 10, 11, 12 |

Identical numeric lap IDs in dry and wet refer to separate condition-specific recordings. There is no random sample-level train/test split, and no dry lap belongs to more than one split. All three classes are present in every split, so the planned 5/1/2 dry-lap assignment was retained.

The resulting future-onset dataset is:

| Split | Eligible windows | Neutral | Understeer | Oversteer |
| --- | --- | --- | --- | --- |
| train | 13176 | 12,508 (94.93%) | 536 (4.07%) | 132 (1.00%) |
| validation | 2491 | 2,350 (94.34%) | 120 (4.82%) | 21 (0.84%) |
| dry_test | 5229 | 4,942 (94.51%) | 251 (4.80%) | 36 (0.69%) |
| wet_test | 20923 | 19,754 (94.41%) | 916 (4.38%) | 253 (1.21%) |

This task is much more imbalanced than the original frame-level classification task. Most Neutral histories are not followed by a confirmed event within 250 ms.

Overlapping positive windows frequently point to the same underlying onset. Distinct eligible onset counts are:

| Split | Understeer onsets | Oversteer onsets |
| --- | --- | --- |
| Dry training | 44 | 15 |
| Dry validation | 11 | 2 |
| Dry test | 23 | 4 |
| Wet test | 89 | 32 |

These are onsets represented by at least one eligible prediction window, not all handling events in the raw recordings. For example, the 36 dry-test Oversteer windows correspond to only four distinct eligible onsets.

## 6. Normalization and class weighting

For each feature, the mean and population standard deviation were calculated using only the generated dry training windows, then saved to `scaler.npz`. The same values normalize validation, dry test and wet test. Frames occurring in multiple overlapping training windows contribute multiple times to these statistics. A near-zero standard deviation is replaced with 1 to avoid division by zero.

Normalization is `z = (x - training_mean) / training_std`.

Weighted cross-entropy uses newly calculated dry-training target counts: `weight_c = number_of_training_windows / (3 × training_count_c)`.

| Target | New class weight |
| --- | --- |
| Neutral | 0.351135 |
| Understeer | 8.194030 |
| Oversteer | 33.272727 |

The weights prioritize rare event targets and affect the trade-off between detecting events and issuing false alarms. They were not copied from the original task, and neither dry test nor wet data contributes to them.

## 7. Architectures and exact training protocol

| Setting | LSTM | Vanilla RNN |
| --- | --- | --- |
| Input | 30 × 12 normalized past observations | Identical |
| Recurrent layers | 2 | 2 |
| Hidden width | 64 | 64 |
| Recurrent cell | LSTM | Tanh RNN |
| Dropout between recurrent layers | 0.3 | 0.3 |
| Dropout before final classifier | 0.3 | 0.3 |
| Output | 3 logits from last recurrent output | Identical |
| Trainable parameters | 53,443 | 13,507 |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.001 |
| Batch size | 32 | 32 |
| Loss | Training-count weighted cross-entropy | Identical |
| Epochs run | 30 per seed | 30 per seed |
| Seeds | 42, 43, 44, 45, 46 | Identical |
| Checkpoint selection | Lowest dry validation weighted loss | Identical |

The original LSTM architecture was retained. Every new model was initialized from scratch. All runs completed 30 epochs: there was no early stopping. The best validation-loss state was saved even when it occurred before epoch 30. Training batches were shuffled with a seeded generator; validation and test batches were not shuffled. Deterministic PyTorch algorithms were enabled and four CPU threads were used. Training ran locally on CPU, not on a GPU or Colab.

The RNN has the same width and depth but fewer parameters. Therefore, this is a fixed-width architecture comparison, not a parameter-matched test that isolates gating independently from capacity.

The completed experiment used Python 3.14.2, PyTorch 2.10.0+cpu, NumPy 2.4.2, pandas 3.0.1 and Matplotlib 3.11.1. Exact versions and raw-data hashes are recorded in the snapshot manifest. Exact numerical replication across different hardware or library builds is not guaranteed.

## 8. Main results

All results are window-level. Accuracy is the fraction of correct targets. Class precision is the fraction of predictions for that class that are correct; recall is the fraction of actual targets for that class detected. F1 balances precision and recall. Macro-F1 gives each of the three classes equal weight. Undefined class precision/recall is set to zero.

The following ± values are sample standard deviations across five seeds, not confidence intervals and not uncertainty over independently collected laps. Accuracy uses percentages; macro-F1 uses a 0–1 scale.

| Model | Test domain | Accuracy (%) | Macro-F1 |
| --- | --- | --- | --- |
| LSTM | dry_test | 86.80 ± 3.46 | 0.494 ± 0.023 |
| LSTM | wet_test | 84.70 ± 5.79 | 0.522 ± 0.044 |
| RNN | dry_test | 85.18 ± 1.74 | 0.503 ± 0.028 |
| RNN | wet_test | 80.10 ± 4.08 | 0.507 ± 0.030 |

Class metrics (0–1 scale, mean ± sample SD across seeds):

| Model | Domain | Class | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| LSTM | dry_test | Neutral | 0.980 ± 0.005 | 0.881 ± 0.041 | 0.928 ± 0.021 |
| LSTM | dry_test | Understeer | 0.268 ± 0.056 | 0.680 ± 0.092 | 0.378 ± 0.042 |
| LSTM | dry_test | Oversteer | 0.117 ± 0.023 | 0.361 ± 0.048 | 0.176 ± 0.029 |
| LSTM | wet_test | Neutral | 0.986 ± 0.006 | 0.855 ± 0.067 | 0.914 ± 0.036 |
| LSTM | wet_test | Understeer | 0.251 ± 0.085 | 0.782 ± 0.105 | 0.368 ± 0.079 |
| LSTM | wet_test | Oversteer | 0.209 ± 0.036 | 0.473 ± 0.135 | 0.284 ± 0.032 |
| RNN | dry_test | Neutral | 0.982 ± 0.003 | 0.860 ± 0.021 | 0.917 ± 0.011 |
| RNN | dry_test | Understeer | 0.239 ± 0.023 | 0.758 ± 0.057 | 0.363 ± 0.023 |
| RNN | dry_test | Oversteer | 0.163 ± 0.063 | 0.417 ± 0.144 | 0.231 ± 0.082 |
| RNN | wet_test | Neutral | 0.990 ± 0.003 | 0.801 ± 0.046 | 0.885 ± 0.026 |
| RNN | wet_test | Understeer | 0.189 ± 0.035 | 0.873 ± 0.063 | 0.308 ± 0.039 |
| RNN | wet_test | Oversteer | 0.240 ± 0.026 | 0.540 ± 0.118 | 0.329 ± 0.040 |

### Every individual trial

Accuracy columns below are percentages. Both domain evaluations use the same dry-selected checkpoint for that model and seed.

| Model | Seed | Selected epoch | Dry accuracy | Wet accuracy | Dry macro-F1 | Wet macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | 42 | 3 | 83.90 | 80.61 | 0.487 | 0.482 |
| LSTM | 43 | 4 | 83.08 | 78.00 | 0.468 | 0.491 |
| LSTM | 44 | 11 | 89.44 | 89.39 | 0.507 | 0.554 |
| LSTM | 45 | 7 | 86.50 | 83.74 | 0.481 | 0.500 |
| LSTM | 46 | 14 | 91.11 | 91.74 | 0.526 | 0.582 |
| RNN | 42 | 2 | 84.45 | 76.28 | 0.480 | 0.494 |
| RNN | 43 | 2 | 84.80 | 78.47 | 0.467 | 0.500 |
| RNN | 44 | 9 | 83.86 | 79.18 | 0.529 | 0.503 |
| RNN | 45 | 25 | 88.24 | 87.05 | 0.519 | 0.559 |
| RNN | 46 | 17 | 84.57 | 79.52 | 0.522 | 0.481 |

### Paired domain changes

Each difference is calculated as dry minus wet for the same seed, then summarized across five seeds. Positive values mean a lower wet score; negative values mean a higher wet score. Accuracy and recall differences are percentage points; macro-F1 differences remain on the 0–1 scale.

| Model | Accuracy change (pp) | Macro-F1 change | Understeer recall change (pp) | Oversteer recall change (pp) |
| --- | --- | --- | --- | --- |
| LSTM | 2.11 ± 2.36 | -0.028 ± 0.024 | -10.26 ± 4.76 | -11.16 ± 14.60 |
| RNN | 5.08 ± 2.57 | -0.004 ± 0.036 | -11.49 ± 5.30 | -12.33 ± 17.64 |

For the LSTM, accuracy declines by about 2.11 percentage points on average while macro-F1 increases by about 0.0285. Understeer recall increases by about 10.26 percentage points and Oversteer recall by about 11.16 points. For the RNN, accuracy declines by about 5.08 points while macro-F1 increases slightly, by about 0.0040.

### Always-Neutral baseline

| Domain | Accuracy | Macro-F1 | Understeer recall | Oversteer recall |
| --- | --- | --- | --- | --- |
| Dry test | 94.51% | 0.324 | 0 | 0 |
| Wet test | 94.41% | 0.324 | 0 | 0 |

An always-Neutral predictor has higher accuracy than either trained architecture because Neutral targets dominate. It detects no events. Both networks have higher macro-F1, but their rare-event precision is low. For example, LSTM wet precision is about 0.251 for Understeer and 0.209 for Oversteer. Many predicted event windows are false alarms under the operational labeling rule.

## 9. Interpretation that the evidence supports

The models show some ability to discriminate future, physics-defined event windows from past Neutral telemetry, as reflected in macro-F1 above the always-Neutral baseline and nonzero event recall. This is evidence for a limited predictive signal in this fixed experiment. It does not establish that the models would be useful warning systems, nor does it isolate the contribution of ordered temporal memory.

The wet results should not be described as a uniform deterioration in generalization. Both models have lower mean wet accuracy, yet higher mean wet event recall and macro-F1. Changing conditions changes the distribution of inputs and prediction errors; it does not necessarily make every metric worse. Differences in the particular laps, eligible event mix, driving behavior and class distributions may also contribute. No causal explanation for the improvement was tested.

Understeer and Oversteer behave differently. LSTM wet recall is approximately 0.782 for Understeer versus 0.473 for Oversteer. Its wet F1 values are approximately 0.368 and 0.284 respectively. Rare-event discrimination is considerably weaker than Neutral classification, and Oversteer remains difficult. Although Oversteer recall and F1 improve from dry to wet, its dry test has only four distinct eligible events, so that comparison is fragile.

There is no clear overall LSTM advantage. LSTM wet macro-F1 is about 0.522 versus RNN 0.507, a mean difference of only about 0.0148. On dry test, the RNN has slightly higher mean macro-F1: 0.503 versus LSTM 0.494. The RNN also has higher mean event recall in both domains, while the LSTM has higher mean accuracy. The paired wet LSTM-minus-RNN macro-F1 differences have a sample SD of about 0.0623, larger than their mean difference. No formal significance test was performed, and five seeds on fixed laps are not sufficient to conclude that gated memory generally outperforms a vanilla RNN.

Moving the target into the future removes the direct concurrent-target problem. However, the model still observes physics variables that may be close to labeling thresholds before onset. That can be legitimate precursor information, but it does not prove that sequence order or long-term memory is responsible for performance. A last-frame-only classifier and a predefined temporal-order ablation were not run. They remain proposed follow-up controls, not completed results.

## 10. Verification that was performed

Synthetic tests exercised a confirmed onset after the input, a single-frame fluctuation, an event outside the horizon, incomplete persistence near the horizon boundary, competing event types with first-onset selection, duplicate timestamps, large gaps, and lap boundaries. The saved-data audit independently recomputed training scaler statistics, checked split membership, verified the 30-frame Neutral histories and timestamp continuity, and re-derived the future targets for all saved dry samples.

The generated sample plot was visually inspected. It shows a Neutral input period, the following 250 ms period, and example Neutral/Understeer/Oversteer targets. The model comparison plot was also inspected.

All ten checkpoints and twenty dry/wet evaluations were checked for completeness. Reloading all ten saved checkpoints with the saved training scaler reproduced their wet confusion matrices exactly. Raw CSV changes were checked with Git: the raw files were unchanged. Code compilation and Git whitespace checks passed.

These checks verify implementation consistency. They do not independently validate the physical correctness of the pre-existing labels. The saved-row audit explicitly covered dry splits; wet evaluation uses the same window-construction function, but a separate exhaustive wet metadata audit was not performed.

## 11. Limitations to include in the essay

1. **Small independent sample size.** Thousands of overlapping windows do not equal thousands of independent events. Dry test contains only two laps and four eligible Oversteer onsets; validation contains only two eligible Oversteer onsets.
2. **Strong imbalance and false alarms.** Accuracy is misleading without event metrics. Weighted loss improves attention to rare classes but does not guarantee high precision or calibrated probabilities.
3. **Conditional eligibility.** Physics labels are used to retain entirely Neutral histories. A deployed system would need its own way to establish that eligibility. This offline experiment does not evaluate that additional stage.
4. **Operational rather than independent ground truth.** Label thresholds and logger implementation were unavailable. Findings are about these saved labels; threshold validity/sensitivity was not tested.
5. **Timing assumptions.** Timestamp units are inferred from lap duration. Thirty frames and three-frame persistence have variable physical durations. The 50 ms gap rule and within-horizon confirmation rule influence which samples and targets exist.
6. **Late-onset convention.** An onset inside the last part of the 250 ms interval may be labeled Neutral if confirmation extends beyond the interval. The implemented target must be stated explicitly rather than claiming all possible starts within 250 ms are counted.
7. **Limited domain coverage.** One recorded car, one track and a small set of simulated dry/wet laps do not establish generalization to other cars, tracks, drivers or real vehicles.
8. **Confounded environmental differences.** Temperature and grip change together, and driving behavior may differ. This design assesses transfer between the recorded conditions without isolating each cause.
9. **Architecture capacity difference.** LSTM has roughly four times the trainable parameters of the same-width RNN. Gating and capacity are not isolated.
10. **No temporal ablation or stronger static baseline.** The always-Neutral baseline is useful for imbalance, but does not establish the value of memory relative to a last-frame classifier. No causal claim about temporal order is supported.
11. **Seed uncertainty is limited.** Mean ± SD across seeds measures optimization variation on a fixed lap split. It is not an independent-lap confidence interval. No bootstrap, formal significance test or event-level detection evaluation was completed.
12. **Wet test is now observed.** Later changes motivated by these wet results would make this dataset part of development history. A stronger confirmatory study would need a predefined follow-up protocol and fresh held-out data; the present runs were not retuned after wet evaluation.

## 12. What changed in the repository

| File | Change or purpose |
| --- | --- |
| `dataset_lstm.py` | Complete replacement of concurrent-label dataset with explicit feature list, fixed lap splits, strict CSV validation, past-only Neutral windows, timestamp-based future-onset labels and external training-scaler application |
| `train_lstm.py` | Complete replacement of random-window split training with reproducible LSTM/RNN trials, training-only weights/scaler, validation-loss checkpoints, frozen dry/wet evaluation and persisted metrics/predictions |
| `testwet_lstm.py` | Complete replacement of legacy wet-only normalization/checkpoint loading with evaluation of new checkpoints using the saved dry-training scaler |
| `model_lstm.py` | Original LSTM architecture retained |
| `model_rnn.py` | New two-layer vanilla tanh RNN baseline |
| `diagnose_samples.py` | New synthetic boundary tests and visual sample audit |
| `audit_pipeline.py` | New independent audit of saved dry metadata and scaler |
| `report_results.py` | New aggregate results, paired domain changes, Neutral baseline, distinct-onset counts and comparison plot |
| `EXPERIMENT_METHOD.md` | Variables, controls, leakage prevention, evaluation and limits |
| `EXPERIMENT_HANDOFF.md` | This complete pasteable explanation |
| `Readme.md` | Revised usage, method and artifact guidance |
| `requirements.txt` | Exact versions of the main installed packages |
| `.gitignore` | Excludes generated rerun directories, logs, bytecode, plotting cache and local virtual environment |
| `reference_results/2026-10-14/` | Preserved completed experiment outputs, included in GitHub |
| `data/dry_laps.csv`, `data/wet_laps.csv` | Unchanged raw telemetry |
| `handling_lstm.pth` | Preserved historical current-state checkpoint; not used for forecasting |

The working run directory remains `results_future/` locally. Its completed contents were copied to the versioned `reference_results/2026-10-14/` directory so the published evidence is preserved when the training command is run again.

The original repository was cloned from commit `e7a34ea08d36052338cb1e1c44ded18e4a68970a`. The experiment snapshot manifest contains data hashes and hashes of the Python source files used for the completed run. Some provenance fields were added after the run during auditing; the training script itself automatically records the split, features, counts, settings, package versions and raw-data hashes. A new run does not automatically add every post-run provenance field.

## 13. Where to find the evidence

All paths below are relative to `reference_results/2026-10-14/`:

- `manifest.json`: actual settings, data identities, split IDs, features, counts, weights and provenance.
- `scaler.npz`: dry-training feature means and standard deviations.
- `metrics.csv`: all twenty model/seed/domain records, including every class metric, selected epoch and parameter count.
- `metrics.json`: the same records plus every per-run confusion matrix.
- `aggregate.csv`: mean and sample SD of accuracy, macro-F1 and all class metrics. Its first rows form a multi-level CSV header.
- `paired_dry_minus_wet.csv` and `paired_gap_summary.csv`: within-seed domain changes and their summaries.
- `neutral_baseline.json`: exact always-Neutral results.
- `distinct_onsets.csv`: eligible event counts for the two test domains.
- `train_samples.csv`, `validation_samples.csv`, `dry_test_samples.csv`, `wet_test_samples.csv`: lap and timing boundaries, target, onset time and sorted within-lap row indices for every sample.
- `{model}_{seed}.pth`: each of the ten selected checkpoints, including model type, seed, selected epoch and parameter count.
- `{model}_{seed}_history.csv`: all 30 training/validation loss records for that run.
- `{model}_{seed}_{domain}_predictions.csv`: target and prediction in the same order as that domain's sample metadata.
- `sample_audit.png`: visual examples of input and future periods.
- `model_comparison.png`: accuracy and macro-F1 with seed SD error bars.
- `RESULTS.md`: shorter results summary.

Confusion-matrix rows are actual classes; columns are predicted classes. Class order is always Neutral, Understeer, Oversteer. The complete per-run matrices are in metrics.json; summed matrices in the appendix below combine repeated evaluations across seeds and must not be interpreted as independent new samples.

## 14. Instructions to inspect and rerun

From PowerShell, clone into a new folder name that does not already exist:

```powershell
git clone https://github.com/GTs5Devy/LSTM_EE.git LSTM_EE_reproduction
Set-Location LSTM_EE_reproduction
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The completed reference results are already included, so retraining is not necessary to inspect them. To reload the published checkpoints and reproduce their wet evaluation:

```powershell
.\.venv\Scripts\python.exe testwet_lstm.py --results reference_results/2026-10-14
```

To prepare and check new datasets without training:

```powershell
.\.venv\Scripts\python.exe train_lstm.py --prepare-only
.\.venv\Scripts\python.exe diagnose_samples.py
.\.venv\Scripts\python.exe audit_pipeline.py
```

To run the full original protocol again and regenerate reports:

```powershell
.\.venv\Scripts\python.exe train_lstm.py
.\.venv\Scripts\python.exe report_results.py
.\.venv\Scripts\python.exe audit_pipeline.py
.\.venv\Scripts\python.exe testwet_lstm.py
```

Run commands from the repository root because data paths are relative. The default training command creates/overwrites files in `results_future/`, not in the published reference snapshot. `report_results.py`, `audit_pipeline.py` and `diagnose_samples.py` currently use the default `results_future/` directory. `testwet_lstm.py` supports an explicit results path.

For a short pipeline smoke test:

```powershell
.\.venv\Scripts\python.exe train_lstm.py --epochs 1 --seeds 42 --out results_smoke
.\.venv\Scripts\python.exe testwet_lstm.py --results results_smoke
```

The smoke command uses both architectures unless `--models` is specified. Its one-epoch numbers are not the final essay evidence. Do not mix them into the five-seed results. Do not run `--prepare-only` in a completed working-results directory merely to inspect it: that command rewrites its preparation files and manifest. Use the preserved reference files for inspection.

If creating a deliberately different protocol, use a new output directory, record the change before testing, and distinguish it from this experiment. Changing splits, thresholds or hyperparameters because of already-seen wet results would weaken the claim that wet remained a final held-out test.

## 15. A conclusion consistent with the completed trials

Under the documented future-onset definition, the dry-trained LSTM achieved a mean macro-F1 of 0.494 on held-out dry laps and 0.522 on unseen wet laps, compared with an always-Neutral macro-F1 of approximately 0.324. These results indicate limited discrimination of future physics-defined handling events from past Neutral telemetry. However, low event precision, particularly for Oversteer, limits the reliability of the forecasts. Transfer to wet conditions produced lower accuracy but higher macro-F1 and event recall, so environmental change did not cause a uniform decline across metrics. The LSTM did not show a consistent advantage over the vanilla RNN: its wet mean macro-F1 was slightly higher, while its dry mean macro-F1 was lower. The small number of independent events, unequal parameter counts and absence of temporal ablations prevent a strong claim that LSTM gated memory caused superior generalization. The revised design is more defensible than concurrent-state classification because future target observations are excluded from model inputs, but its conclusions remain conditional on the saved physics labels, eligibility rules, fixed lap split and single simulation setting.

## Appendix: confusion matrices summed across the five seeds

These sums describe the pattern of errors across five repeated model runs. Each underlying test window is counted five times. They are not event counts, independent observations, or confidence estimates. Rows = actual and columns = predicted; class order = Neutral, Understeer, Oversteer.

### LSTM — dry_test

| Actual / predicted | Neutral | Understeer | Oversteer |
| --- | --- | --- | --- |
| Neutral | 21777 | 2504 | 429 |
| Understeer | 329 | 853 | 73 |
| Oversteer | 115 | 0 | 65 |

### LSTM — wet_test

| Actual / predicted | Neutral | Understeer | Oversteer |
| --- | --- | --- | --- |
| Neutral | 84425 | 12096 | 2249 |
| Understeer | 910 | 3583 | 87 |
| Oversteer | 328 | 339 | 598 |

### RNN — dry_test

| Actual / predicted | Neutral | Understeer | Oversteer |
| --- | --- | --- | --- |
| Neutral | 21245 | 3061 | 404 |
| Understeer | 292 | 951 | 12 |
| Oversteer | 105 | 0 | 75 |

### RNN — wet_test

| Actual / predicted | Neutral | Understeer | Oversteer |
| --- | --- | --- | --- |
| Neutral | 79116 | 17552 | 2102 |
| Understeer | 514 | 3997 | 69 |
| Oversteer | 273 | 309 | 683 |
