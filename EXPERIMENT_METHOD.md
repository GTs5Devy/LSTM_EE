# Experimental method for the revised Extended Essay

## Scope

The task is a three-class forecast of physics-defined handling onset, conditional on an entirely Neutral recent history. Dry telemetry is the training domain. Held-out dry laps measure generalization to other laps in the same condition; wet laps measure transfer to a changed environmental condition. This distinction prevents the wet score from being mistaken for a training score.

## Variables and controls

| Role | Operational definition |
|---|---|
| Environmental comparison | Dry held-out versus wet unseen telemetry |
| Architecture comparison | Two-layer LSTM versus two-layer vanilla tanh RNN; optional last-frame MLP control |
| Primary dependent measures | Macro-F1 and class precision/recall; accuracy is supplementary |
| Fixed inputs | Same 12 physics features, 30-frame history, 250 ms future interval |
| Fixed target rule | First event transition beginning within 250 ms, confirmed by three consecutive matching observations; confirmation may extend after the horizon |
| Fixed training protocol | Same whole-lap split, dry-training scaler and weights, Adam 0.001, batch 32, 30 epochs, validation-loss checkpoint selection |
| Replication | Seeds 42,43,44,45,46; same seed IDs paired between architectures |
| Simulation controls | Car and track as recorded in the CSVs; confirm their identifiers match between domains |

An event is a future change to Understeer or Oversteer in the existing Label column. Neutral means no confirmed onset beginning within the horizon, not necessarily that every future observation is Neutral. Single-frame fluctuations are not positives. The next two matching observations may occur after 250 ms solely to confirm an onset that began inside the interval. Inputs remain 30 consecutive telemetry observations (~0.5 s), not exactly 0.5 seconds.

## Preventing leakage

Whole laps are assigned before overlapping windows are constructed. No lap appears in more than one dry split. Inputs and target observations do not overlap. Every input label is Neutral, and Label itself is excluded from model features. Exclude YawGradient, position, time, car/track identifiers and environmental measurements from features. The training scaler and class weights use only dry training windows. Validation selects checkpoints without wet results; no wet statistics normalize the wet data.

Sorting is done separately within each lap. A nonpositive timestamp increment or a gap greater than 50 ms invalidates any window spanning it, including the required post-horizon confirmation allowance. Full coverage is required even for Neutral candidates. This preserves temporal continuity without inventing interpolated measurements. The recovered logger records graphics.iCurrentTime; timestamps are interpreted as milliseconds, consistent with the recorded lap durations.

## Evaluation and uncertainty

Use class order Neutral, Understeer, Oversteer for every report. Confusion matrices have actual classes in rows and predictions in columns. Macro-F1 averages the three class F1 values equally. Event recall is the fraction of positive target windows recognized for that event class. Precision concerns the reliability of predicted event windows. These are window-level, not independent event-level, measures.

Report mean and sample standard deviation across five seeds. Subtract wet from dry separately for each seed before summarizing transfer loss. A positive difference indicates worse wet performance; a negative difference indicates better wet performance. Seed standard deviations reflect optimization variability on this fixed dataset, not uncertainty over cars, tracks, drivers or independent data collection.

Report an always-Neutral baseline: high accuracy can follow directly from the large Neutral class. Distinct eligible onset counts accompany window counts. Many nearby windows predict the same onset, and should not be described as separate independent trials. Optional event-level evaluation counts each represented onset once and reports false-warning episodes per lap and per eligible monitoring minute. Its exact predeclared definitions and denominator restrictions are in PROTOCOL_V2.md and event_protocol.json.

## Limits on conclusions

The preserved labels are operational ground truth, not independently measured physical truth. The installed logger was recovered and its rule reproduces all audited dry/wet interior labels and YawGradient values. The older user-supplied GitHub logger differs and is preserved for provenance. See PHYSICS_LABELER.md for exact rules and exclusions. This verifies implementation consistency, not physical validity or threshold sensitivity. Raw labels were not regenerated. A changed task invalidates the old current-state classification results as evidence for forecasting.

Removing concurrent target leakage makes a precursor interpretation possible, but strong performance does not prove reliance on ordered memory. The optional last-frame MLP compares these sequence models against a fixed static control, without isolating architecture capacity. A temporal-order ablation remains unperformed. LSTM and RNN use the same hidden width and depth but have different parameter counts, so an advantage cannot be attributed exclusively to gating. The recovered labels themselves use a temporal history, which also limits interpreting this operational task as independent physical evidence.

The study uses few laps, sparse Oversteer onsets, one simulation/car/track combination, and existing condition-specific driving behavior. Environmental changes and differences in driver behavior may both contribute to domain shift. Any result is specific to the physics-defined target, eligibility filter, data and timing choices documented here.
