# Experimental method for the revised Extended Essay

## Scope

The task is a three-class forecast of physics-defined handling onset, conditional on an entirely Neutral recent history. Dry telemetry is the training domain. Held-out dry laps measure generalization to other laps in the same condition; wet laps measure transfer to a changed environmental condition. This distinction prevents the wet score from being mistaken for a training score.

## Variables and controls

| Role | Operational definition |
|---|---|
| Environmental comparison | Dry held-out versus wet unseen telemetry |
| Architecture comparison | Two-layer LSTM versus two-layer vanilla tanh RNN |
| Dependent measures | Macro-F1, class precision/recall/F1, accuracy, confusion matrices and dry-minus-wet differences |
| Fixed inputs | Same 12 physics features, 30-frame history, 250 ms future interval |
| Fixed target rule | First event transition with three consecutive matching observations within the future interval |
| Fixed training protocol | Same whole-lap split, dry-training scaler and weights, Adam 0.001, batch 32, 30 epochs, validation-loss checkpoint selection |
| Replication | Seeds 42,43,44,45,46; same seed IDs paired between architectures |
| Simulation controls | Car and track as recorded in the CSVs; confirm their identifiers match between domains |

An event is a future change to Understeer or Oversteer in the existing Label column. Neutral means no event meeting the confirmation rule within the horizon, not necessarily that every future observation is Neutral. In particular, single-frame fluctuations and late events without sufficient confirmation are not positive targets.

## Preventing leakage

Whole laps are assigned before overlapping windows are constructed. No lap appears in more than one dry split. Inputs and target observations do not overlap. Every input label is Neutral, and Label itself is excluded from model features. Exclude YawGradient, position, time, car/track identifiers and environmental measurements from features. The training scaler and class weights use only dry training windows. Validation selects checkpoints without wet results; no wet statistics normalize the wet data.

Sorting is done separately within each lap. A nonpositive timestamp increment or a gap greater than 50 ms invalidates any window spanning it. This preserves temporal continuity without inventing interpolated measurements. The timestamp unit is interpreted as milliseconds from the recorded lap durations; the logger implementation is absent and this interpretation should be checked against collection notes.

## Evaluation and uncertainty

Use class order Neutral, Understeer, Oversteer for every report. Confusion matrices have actual classes in rows and predictions in columns. Macro-F1 averages the three class F1 values equally. Event recall is the fraction of positive target windows recognized for that event class. Precision concerns the reliability of predicted event windows. These are window-level, not independent event-level, measures.

Report mean and sample standard deviation across five seeds. Subtract wet from dry separately for each seed before summarizing transfer loss. A positive difference indicates worse wet performance; a negative difference indicates better wet performance. Seed standard deviations reflect optimization variability on this fixed dataset, not uncertainty over cars, tracks, drivers or independent data collection.

Report an always-Neutral baseline: high accuracy can follow directly from the large Neutral class. Distinct eligible onset counts accompany window counts. Many nearby windows predict the same onset, and should not be described as separate independent trials.

## Limits on conclusions

The preserved labels are operational ground truth, not independently measured physical truth. Their original generation code is not included, so the experiment cannot verify their implementation or sensitivity to thresholds. A changed task invalidates the old current-state classification results as evidence for forecasting.

Removing concurrent target leakage makes a precursor interpretation possible, but strong performance does not prove reliance on ordered memory. A last-frame baseline or a predefined temporal-order ablation would be necessary to strengthen that particular claim. LSTM and RNN use the same hidden width and depth but have different parameter counts, so an advantage cannot be attributed exclusively to gating.

The study uses few laps, sparse Oversteer onsets, one simulation/car/track combination, and existing condition-specific driving behavior. Environmental changes and differences in driver behavior may both contribute to domain shift. Any result is specific to the physics-defined target, eligibility filter, data and timing choices documented here.
