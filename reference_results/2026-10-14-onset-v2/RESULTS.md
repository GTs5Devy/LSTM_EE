# Future-onset trial results

Five seeds; error bars and standard deviations describe seed variation, not independent lap uncertainty.

```
                     macro_f1           Understeer_precision           Understeer_recall           Oversteer_precision           Oversteer_recall            accuracy
                         mean       std                 mean       std              mean       std                mean       std             mean       std      mean       std
model      domain
last_frame dry_test  0.507134  0.012409             0.275158  0.012800          0.834364  0.048925            0.126336  0.017268         0.460000  0.048734  0.841852  0.013471
           wet_test  0.496405  0.015768             0.193899  0.017882          0.928369  0.028432            0.209854  0.011664         0.582578  0.043407  0.766359  0.024518
lstm       dry_test  0.542974  0.039078             0.318973  0.033111          0.632990  0.124790            0.204642  0.094178         0.550000  0.259808  0.879572  0.034205
           wet_test  0.553119  0.032158             0.296996  0.038895          0.698021  0.081815            0.235598  0.054982         0.558188  0.164565  0.867664  0.030911
rnn        dry_test  0.495820  0.031059             0.258352  0.015640          0.830928  0.118180            0.122075  0.057513         0.465000  0.307002  0.831140  0.022153
           wet_test  0.503832  0.032068             0.195083  0.027607          0.886145  0.077152            0.235441  0.041514         0.574216  0.127580  0.774105  0.048766
```

Always-Neutral baselines:

```
  domain  accuracy  macro_f1
dry_test  0.936687  0.322436
wet_test  0.935549  0.322234
```

Distinct eligible onsets:

```
  domain  windows  understeer_onsets  oversteer_onsets
dry_test     5228                 23                 4
wet_test    20915                 90                33
```

An onset must start within 250 ms after 30 consecutive Neutral telemetry observations (~0.5 s). The following two matching observations may extend beyond 250 ms solely for persistence confirmation. Labels are recovered logger-derived CSV labels. Macro-F1 and class precision/recall are primary; accuracy is supplementary. Forecasts remain conditional on Neutral-history eligibility. Same-width LSTM/RNN comparisons do not control parameter count.

lstm: dry macro-F1 0.543, wet 0.553; dry accuracy 0.880, wet 0.868.

rnn: dry macro-F1 0.496, wet 0.504; dry accuracy 0.831, wet 0.774.

last_frame: dry macro-F1 0.507, wet 0.496; dry accuracy 0.842, wet 0.766.

Optional last-frame control: paired sequence-minus-last-frame macro-F1:

```
                    mean       std
model domain
lstm  dry_test  0.035841  0.047804
      wet_test  0.056714  0.035132
rnn   dry_test -0.011313  0.024836
      wet_test  0.007426  0.043786
```

This control tests the value of these sequence models relative to this static MLP. Different architectures/capacities prevent a causal memory-only conclusion.

Optional event-level evaluation:

```
                               event_recall           false_episodes_per_lap            false_episodes_per_eligible_minute
                                       mean       std                   mean        std                               mean        std
model      domain   class
last_frame dry_test Oversteer      0.950000  0.111803              11.700000   2.564176                          16.239287   3.559008
                    Understeer     0.956522  0.000000              27.300000   3.718198                          37.891669   5.160760
           wet_test Oversteer      0.787879  0.047913              12.155556   0.696552                          18.684246   1.070667
                    Understeer     0.931111  0.004969              27.333333   4.630148                          42.013936   7.116979
lstm       dry_test Oversteer      0.700000  0.209165               3.300000   2.361144                           4.580312   3.277204
                    Understeer     0.826087  0.081340              10.200000   2.109502                          14.157327   2.927933
           wet_test Oversteer      0.696970  0.145328               5.022222   2.441159                           7.719634   3.752294
                    Understeer     0.804444  0.061162              12.222222   1.596679                          18.786719   2.454247
rnn        dry_test Oversteer      0.750000  0.176777               5.800000   1.204159                           8.050245   1.671341
                    Understeer     0.973913  0.038888              18.900000   7.478302                          26.232694  10.379683
           wet_test Oversteer      0.739394  0.079021               7.600000   2.221667                          11.681924   3.414913
                    Understeer     0.911111  0.022222              21.755556  11.043796                          33.440361  16.975366
```

Event recall counts distinct eligible first-confirmed onsets, not all raw events. False-warning episodes merge consecutive same-class warnings at <=50 ms gaps; an episode is false if it contains no correct target warning. Rates are conditional on eligible monitoring. See event_protocol.json and raw/eligible event counts.
