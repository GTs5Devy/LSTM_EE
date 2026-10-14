"""Past-only inputs; future labels never enter model features."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
LABEL_MAP = {'Neutral': 0, 'Understeer': 1, 'Oversteer': 2}
FEATURES = ['YawRate','LateralAccel','LongitudinalAccel','SteerAngle','Speed','WheelSlipFL','WheelSlipFR','WheelSlipRL','WheelSlipRR','Throttle','Brake','SlipDiff']
SPLITS = {'train':[1,2,3,4,6], 'validation':[8], 'dry_test':[9,11]}
WINDOW_CONFIG = {'sequence_length': 30, 'horizon_ms': 250, 'persistence': 3,
                 'max_gap_ms': 50, 'confirmation_policy': 'onset_within_horizon'}
def read_csv(path):
    d = pd.read_csv(path, sep=';')
    if not d.Label.isin(LABEL_MAP).all() or not np.isfinite(d[FEATURES+['CurrentTime','Lap']].to_numpy()).all():
        raise ValueError('Invalid telemetry/labels; cannot silently bridge missing rows')
    return d

def build_windows(df, laps, sequence_length=30, horizon_ms=250, persistence=3,
                  max_gap_ms=50, confirmation_policy='onset_within_horizon'):
    if confirmation_policy not in ('onset_within_horizon', 'within_horizon'):
        raise ValueError('Unknown confirmation policy')
    windows, targets, metadata = [], [], []
    for lap in laps:
        d = df[df.Lap == lap].sort_values('CurrentTime', kind='stable')
        x = d[FEATURES].to_numpy(dtype=np.float32)
        y = d.Label.map(LABEL_MAP).to_numpy()
        t = d.CurrentTime.to_numpy()
        for end in range(sequence_length-1, len(d)-1):
            start = end-sequence_length+1
            boundary = np.searchsorted(t, t[end]+horizon_ms, side='right')
            # Reserve enough future observations to confirm an onset at the last
            # in-horizon row. Require the same coverage for Neutral candidates.
            coverage_end = boundary if confirmation_policy == 'within_horizon' else max(boundary, boundary+persistence-2)
            if coverage_end >= len(t): continue
            delta = np.diff(t[start:coverage_end+1])
            if np.any(delta <= 0) or np.any(delta > max_gap_ms) or np.any(y[start:end+1] != 0): continue
            target, onset, confirm_end = 0, None, None
            onset_stop = boundary-persistence+1 if confirmation_policy == 'within_horizon' else boundary
            for j in range(end+1, onset_stop):
                if y[j] != 0 and y[j-1] != y[j] and np.all(y[j:j+persistence] == y[j]):
                    target, onset, confirm_end = int(y[j]), float(t[j]), float(t[j+persistence-1])
                    break
            assert t[end+1] > t[end]
            windows.append(x[start:end+1]); targets.append(target)
            metadata.append(dict(lap=int(lap), input_start_ms=float(t[start]), input_end_ms=float(t[end]), future_end_ms=float(t[end]+horizon_ms), onset_ms=onset, target=target, input_start_row=start, input_end_row=end, future_stop_row=int(boundary), confirmation_stop_row=int(coverage_end+1), confirmation_end_ms=confirm_end))
    if not windows: raise ValueError(f'No eligible windows for {laps}')
    return np.stack(windows), np.array(targets,dtype=np.int64), metadata

class DrivingTimeSeriesDataset(Dataset):
    def __init__(self, windows, labels, mean, std):
        self.sequences = torch.from_numpy(((windows-mean)/std).astype(np.float32))
        self.labels = torch.from_numpy(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.sequences[i], self.labels[i]
