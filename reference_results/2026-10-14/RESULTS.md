# Future-onset trial results

Five seeds; error bars and standard deviations describe seed variation, not independent lap uncertainty.

```
                accuracy            macro_f1           Understeer_recall           Oversteer_recall
                    mean       std      mean       std              mean       std             mean       std
model domain
lstm  dry_test  0.868044  0.034616  0.493664  0.022801          0.679681  0.092435         0.361111  0.048113
      wet_test  0.846972  0.057889  0.522123  0.043780          0.782314  0.104703         0.472727  0.134637
rnn   dry_test  0.851826  0.017430  0.503341  0.027737          0.757769  0.056512         0.416667  0.144338
      wet_test  0.800994  0.040826  0.507367  0.030134          0.872707  0.063223         0.539921  0.117870
```

Always-Neutral baselines:

```
  domain  accuracy  macro_f1
dry_test  0.945114  0.323928
wet_test  0.944128  0.323754
```

Distinct eligible onsets:

```
  domain  windows  understeer_onsets  oversteer_onsets
dry_test     5229                 23                 4
wet_test    20923                 89                32
```

Targets require three consecutive event observations within 250 ms of a 30-frame, entirely Neutral input. Labels are existing physics-derived CSV labels. The forecast is conditional on a known Neutral history; live use would need an independent way to determine that eligibility. Same-width LSTM/RNN comparisons do not control parameter count. These trials remove concurrent-target leakage but cannot alone establish dependence on temporal order.

In these runs, both models have lower mean accuracy on wet data but higher mean macro-F1 and event recall. Thus the measured domain shift is metric-dependent and should not be described as a uniform degradation. Both models outperform the always-Neutral macro-F1 baseline, while having lower accuracy. The LSTM has higher wet mean macro-F1 than the RNN, but lower dry mean macro-F1; five seeds on fixed laps do not establish a general architectural advantage.
