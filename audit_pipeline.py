"""Independently audit saved dry sample boundaries and preprocessing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dataset_lstm import FEATURES, SPLITS, LABEL_MAP, read_csv, build_windows

def main():
    out=Path('results_future'); d=read_csv('data/dry_laps.csv'); scaler=np.load(out/'scaler.npz')
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['splits']==SPLITS and manifest['features']==FEATURES
    training=build_windows(d,SPLITS['train'])[0]
    np.testing.assert_allclose(scaler['mean'],training.mean((0,1),dtype=np.float64))
    s=training.std((0,1),dtype=np.float64); s[s<1e-8]=1
    np.testing.assert_allclose(scaler['std'],s)
    for split,laps in SPLITS.items():
        samples=pd.read_csv(out/(split+'_samples.csv'))
        assert set(samples.lap).issubset(laps)
        for lap,group in samples.groupby('lap'):
            z=d[d.Lap==lap].sort_values('CurrentTime',kind='stable').reset_index(drop=True)
            y=z.Label.map(LABEL_MAP).to_numpy(); t=z.CurrentTime.to_numpy()
            for row in group.itertuples():
                start,end,stop=int(row.input_start_row),int(row.input_end_row),int(row.future_stop_row)
                assert end-start+1==30 and np.all(y[start:end+1]==0)
                assert row.input_end_ms==t[end] and row.input_start_ms==t[start]
                assert t[end+1]>t[end] and t[stop]>t[end]+250 and t[stop-1]<=t[end]+250
                assert np.all((np.diff(t[start:stop+1])>0)&(np.diff(t[start:stop+1])<=50))
                expected=0
                for j in range(end+1,stop-2):
                    if y[j]!=0 and y[j-1]!=y[j] and np.all(y[j:j+3]==y[j]):
                        expected=int(y[j]); assert row.onset_ms==t[j]; break
                assert expected==row.target
        unique=samples[samples.target!=0].drop_duplicates(['lap','onset_ms','target'])
        print(split,'unique eligible onsets',unique.groupby('target').size().to_dict())
    print('Saved split, scaler, input-neutrality, gap and future-only target audit passed')
if __name__=='__main__': main()
