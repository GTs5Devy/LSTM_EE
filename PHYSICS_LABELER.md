# Recovered physics-label implementation

The labeler was recovered from the installed Assetto Corsa app at `apps/python/LapTimeML/LapTimeML.py`. Its byte-preserved copy is `labeling_source/LapTimeML_installed_v3.py`. The user-supplied repository, https://github.com/GTs5Devy/Assetto_Corsa_Telemetry_Extraction, was also inspected at commit `1a76226bebad89aff838bb030d623078d9caeebf`; its older copy is `labeling_source/LapTimeML_github_v2.py`.

These versions differ. The installed copy uses a 20-observation history and filters absolute yaw above 2.0. The GitHub copy uses a 5-observation history and filters absolute yaw above 1.0. Both are preserved rather than silently treating the GitHub version as the source of the current CSVs.

## Verification evidence

`verify_labeler.py` extracts the existing function through Python AST and replays it without importing or executing the live Assetto Corsa app. No thresholds are fitted to the data. Recorded row order is retained. The first 20 rows of each retained lap are excluded from the agreement claim because the logger clears histories after handling the first row of a lap and clean-lap selection can remove preceding context.

| Source | Domain | Audited interior rows | Matching Label | Matching YawGradient |
| --- | --- | --- | --- | --- |
| Installed v3 | Dry | 41,592 | 41,592 | 41,592 |
| Installed v3 | Wet | 47,097 | 47,097 | 47,097 |
| GitHub v2 | Dry | 41,592 | 39,915 | 814 |
| GitHub v2 | Wet | 47,097 | 45,603 | 561 |

The rear-minus-front SlipDiff formula matches every raw row in both files: 41,752 dry rows and 47,277 wet rows, using the stated numerical tolerances. The installed implementation therefore has strong empirical provenance support for these CSVs. This is implementation verification, not independent validation that the rules measure physical understeer/oversteer correctly. Warmup/boundary rows are not claimed as independently reconstructed.

Source hashes, counts and disagreement diagnostics are saved by `verify_labeler.py` in the new run directory. The original CSV Labels and YawGradient are unchanged. The recovered function is used for verification/documentation only; it does not relabel the forecasting dataset.

## Exact installed rules, in execution order

The logger reads YawRate from `physics.localAngularVel[2]`, LateralAccel from `physics.accG[0]`, SteerAngle from `physics.steerAngle`, and Speed from `physics.speedKmh`. It computes:

`SlipDiff = (WheelSlipRL + WheelSlipRR)/2 - (WheelSlipFL + WheelSlipFR)/2`.

The current observation is appended to the history buffers before the label function runs. The buffers retain 20 observations. Their physical duration varies with sampling.

1. Return Neutral and gradient 0 if speed <50, or the yaw/lateral buffers do not contain 20 observations.
2. Calculate `yaw_change = yaw_last - yaw_first`, `lateral_change = lateral_last - lateral_first`, and `speed_change = speed_last - speed_first` across those buffers.
3. Return Neutral and gradient 0 if `abs(yaw_rate)>2.0` or `speed_change<-15`. These are the implementation's spin/deceleration filters; filtered observations are labeled Neutral rather than assigned a separate invalid/spin class.
4. If `abs(lateral_change)<0.01`, use the fallback branch: return Neutral when `abs(lateral_accel)<0.3`; otherwise return Oversteer for `SlipDiff>0.15`, Understeer for `SlipDiff<-0.15`, or Neutral. Gradient is 0 in this entire branch.
5. Otherwise calculate `YawGradient = yaw_change/lateral_change`.
6. Set a speed factor of 0.3 below 80, 0.5 below 120, and 0.7 otherwise. Calculate `expected_yaw = abs(steer_angle)*(speed/100)*speed_factor`.
7. If expected_yaw >0.05, calculate `response_ratio=abs(yaw_rate)/expected_yaw`. If response_ratio >1.5 OR YawGradient >1.0, return Oversteer only if SlipDiff >0.10. The mutually exclusive `elif` branch tests response_ratio <0.6 OR YawGradient <0.15 and returns Understeer only if SlipDiff <-0.10. Preserve this if/elif priority: failure of the slip test in the first branch does not enable the second branch.
8. If no previous branch returns and `abs(lateral_accel)>0.5` and speed >80, return Oversteer for SlipDiff >0.25 or Understeer for SlipDiff <-0.25.
9. Otherwise return Neutral. From step 5 onward the calculated gradient is returned regardless of class.

The source comments describe the rules as Bergman-based, but the empirical thresholds and expected-yaw expression have not been independently calibrated or established as a direct derivation from the cited physical theory. In the essay, distinguish the recovered operational implementation from external physical validation. The app also encourages collecting different handling states, so the recordings are controlled experimental data rather than an estimate of naturally occurring event prevalence.
