"""Independently audit saved dry sample boundaries and preprocessing."""
import json
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from dataset_lstm import FEATURES, SPLITS, LABEL_MAP, read_csv, build_windows
from experiment_config import DEFAULT_RESULTS, window_config

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',default=DEFAULT_RESULTS); args=ap.parse_args()
    out=Path(args.results); d=read_csv('data/dry_laps.csv'); scaler=np.load(out/'scaler.npz')
    manifest=json.loads((out/'manifest.json').read_text())
    config=window_config(manifest); horizon=config['horizon_ms']; persistence=config['persistence']
    assert manifest['splits']==SPLITS and manifest['features']==FEATURES
    training,train_labels,_=build_windows(d,SPLITS['train'],**config)
    np.testing.assert_allclose(scaler['mean'],training.mean((0,1),dtype=np.float64))
    s=training.std((0,1),dtype=np.float64); s[s<1e-8]=1
    np.testing.assert_allclose(scaler['std'],s)
    counts=np.bincount(train_labels,minlength=3)
    np.testing.assert_allclose(manifest['class_weights'],counts.sum()/(3*counts))
    assert set(sum(SPLITS.values(),[]))==set(d.Lap.unique())
    assert len(sum(SPLITS.values(),[]))==len(set(sum(SPLITS.values(),[])))
    sources=[(split,laps,d) for split,laps in SPLITS.items()]
    if 'wet_test' in manifest:
        sources.append(('wet_test',manifest['wet_test']['laps'],read_csv('data/wet_laps.csv')))
    for split,laps,source in sources:
        samples=pd.read_csv(out/(split+'_samples.csv'))
        assert set(samples.lap).issubset(laps)
        _,labels,metadata=build_windows(source,laps,**config)
        np.testing.assert_array_equal(samples.target,labels)
        np.testing.assert_array_equal(samples.lap,[r['lap'] for r in metadata])
        assert len(samples)==len(metadata)
        for lap,group in samples.groupby('lap'):
            z=source[source.Lap==lap].sort_values('CurrentTime',kind='stable').reset_index(drop=True)
            y=z.Label.map(LABEL_MAP).to_numpy(); t=z.CurrentTime.to_numpy()
            for row in group.itertuples():
                start,end,stop=int(row.input_start_row),int(row.input_end_row),int(row.future_stop_row)
                assert end-start+1==config['sequence_length'] and np.all(y[start:end+1]==0)
                assert row.input_end_ms==t[end] and row.input_start_ms==t[start]
                assert t[end+1]>t[end] and t[stop]>t[end]+horizon and t[stop-1]<=t[end]+horizon
                coverage_stop=int(getattr(row,'confirmation_stop_row',stop+1))
                assert np.all((np.diff(t[start:coverage_stop])>0)&(np.diff(t[start:coverage_stop])<=config['max_gap_ms']))
                expected=0
                onset_stop=stop-persistence+1 if config['confirmation_policy']=='within_horizon' else stop
                for j in range(end+1,onset_stop):
                    if y[j]!=0 and y[j-1]!=y[j] and np.all(y[j:j+persistence]==y[j]):
                        expected=int(y[j]); assert row.onset_ms==t[j]; break
                assert expected==row.target
                if expected:
                    assert 0 < row.onset_ms-t[end] <=horizon
                    if hasattr(row,'confirmation_end_ms'): assert row.confirmation_end_ms==t[j+persistence-1]
        unique=samples[samples.target!=0].drop_duplicates(['lap','onset_ms','target'])
        print(split,'unique eligible onsets',unique.groupby('target').size().to_dict())
    for path,digest in manifest['sha256'].items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    for name,digest in manifest.get('checkpoint_sha256',{}).items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
    if manifest.get('status')=='complete':
        metrics=pd.read_csv(out/'metrics.csv')
        models=manifest['args']['models']; seeds=manifest['args']['seeds']; epochs=manifest['args']['epochs']
        assert len(metrics)==len(models)*len(seeds)*2
        assert metrics.groupby(['model','seed','domain']).size().eq(1).all()
        for kind in models:
            for seed in seeds:
                assert len(pd.read_csv(out/f'{kind}_{seed}_history.csv'))==epochs
                for domain in ['dry_test','wet_test']:
                    prediction=pd.read_csv(out/f'{kind}_{seed}_{domain}_predictions.csv')
                    saved=pd.read_csv(out/f'{domain}_samples.csv')
                    np.testing.assert_array_equal(prediction.target,saved.target)
    print('Split, scaler, weights, dry/wet input-neutrality, gaps, future targets and artifacts audit passed')
if __name__=='__main__': main()
