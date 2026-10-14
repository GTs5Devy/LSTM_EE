# Predeclared onset-v2 experiment

The computational settings below were fixed before new wet-model evaluation. They are implemented in the source files hashed in the run manifest and in `event_protocol.json`, which is written before training.

## Required correction

An event must start strictly after the input and no later than 250 ms after its end. Its following two matching observations may lie after the horizon solely to confirm persistence. No onset after the horizon is positive. Three consecutive matching observations are still required. First confirmed onset wins.

The candidate requires enough same-lap data to confirm an onset beginning at the final in-horizon observation. The continuity check extends over that confirmation allowance. The same coverage requirement applies to all candidates, including Neutral ones, so truncation is not treated as evidence of no event. Inputs remain 30 consecutive Neutral observations (~0.5 s). Future observations never enter model features.

LSTM and RNN architectures, 12 features, dry whole-lap split, seeds 42-46, Adam 0.001, batch 32, dropout 0.3, weighted loss, 30 epochs and validation-loss checkpoint selection remain fixed. Models are retrained from scratch with the corrected targets. Normalization and weights are recalculated from corrected dry training windows only.

## Optional control selected before testing

The last-frame model is a one-hidden-layer MLP: Linear(12,64), ReLU, Dropout(0.3), Linear(64,3), 1,027 parameters. It receives only the final past observation. It uses exactly the same eligible examples, split, training scaler, class weights, seeds and training procedure. No width/dropout/architecture search is performed. This control compares the chosen sequence models with this static model; unequal capacities prevent a pure causal memory claim.

## Optional event evaluation selected before testing

Event recall counts distinct first-confirmed onsets represented by at least one eligible positive target window. An event is detected if any such window predicts its matching class before onset, within the preceding 250 ms. Each onset counts once despite overlapping windows. All raw confirmed onsets are also counted, so omitted/unforecastable events are visible.

Warnings are hard argmax event classes; there is no tuned probability threshold. A warning episode is a sequence of consecutive eligible windows with the same event prediction, same lap, and positive end-time gaps <=50 ms. Neutral predictions break episodes. An episode is false when none of its windows correctly predicts the first confirmed future onset. This episode rule can merge early false windows with a later correct warning; per-window false counts are also retained, and episode rates must not be interpreted as window precision.

Report false episodes per lap, per full recorded minute, and per eligible-monitoring minute. Eligible-monitoring time is the sum of <=50 ms gaps between adjacent eligible input-end times in the same lap. Rates are conditional on the Neutral-history monitoring filter and do not simulate continuous deployment. First warning lead time is reported descriptively; no warning threshold is selected using wet results.

## Metrics and limitations

Macro-F1 and class precision/recall are primary. Accuracy is supplementary. Report mean and sample SD over seeds and paired dry-minus-wet changes. Event-level summaries are optional secondary evidence, not independent-lap confidence intervals. Same-width RNN/LSTM parameter differences and grip/temperature confounding remain limitations rather than triggers for tuning. No parameter-matched architecture search, temporal shuffling ablation, or wet-based threshold optimization is included.

The installed logger has been recovered and replay-verified; see PHYSICS_LABELER.md. Saved raw CSV labels remain operational ground truth. Current wet results have already been seen in the previous experiment, so this is a prescribed protocol correction on reused test data, not a newly collected pristine holdout. No settings are chosen from the new wet performance.

## Archive protection

`reference_results/2026-10-14/` is untouched. Training, plotting, report generation, labeler verification and event-evaluation commands reject output paths anywhere under reference_results. Read-only legacy checkpoint evaluation uses the archived confirmation policy from its manifest. Default new outputs use results_onset_v2; existing checkpoint/metrics outputs cannot be overwritten by a training rerun.
