"""Optional, predeclared event summary of the frozen window predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dataset_lstm import LABEL_MAP, read_csv
from experiment_config import DEFAULT_RESULTS, writable_output

EVENT_PROTOCOL = {
    'version': 1,
    'eligible_event': 'distinct first-confirmed onset represented by a positive eligible window target',
    'detected_event': 'at least one matching-class prediction in an eligible window ending strictly before onset and at most 250 ms earlier',
    'warning': 'argmax Understeer or Oversteer; no probability threshold',
    'warning_episode': 'consecutive eligible windows of same non-Neutral predicted class, same lap, end-time gap <=50 ms; Neutral prediction breaks episode',
    'false_warning_episode': 'episode containing no correct prediction of the first confirmed future onset',
    'eligible_exposure': 'sum of end-time increments <=50 ms between adjacent eligible windows in the same lap',
    'rates': 'false episodes per recorded lap, per recorded minute, and per eligible-monitoring minute',
    'max_gap_ms': 50,
    'horizon_ms': 250,
    'persistence': 3,
    'scope': 'conditional on Neutral-history eligibility; full raw onsets are counted separately, not silently treated as forecastable'
}

def confirmed_raw_onsets(df):
    rows=[]
    for lap,z in df.groupby('Lap',sort=True):
        z=z.sort_values('CurrentTime',kind='stable')
        y=z.Label.map(LABEL_MAP).to_numpy(); t=z.CurrentTime.to_numpy()
        for j in range(1,len(z)-2):
            delta=np.diff(t[j-1:j+3])
            if y[j] and y[j]!=y[j-1] and np.all(y[j:j+3]==y[j]) and np.all((delta>0)&(delta<=50)):
                rows.append({'lap':int(lap),'onset_ms':float(t[j]),'target':int(y[j])})
    return pd.DataFrame(rows,columns=['lap','onset_ms','target'])

def evaluate_events(samples, predictions, raw, lap_ids):
    s=samples.reset_index(drop=True).copy(); p=np.asarray(predictions,dtype=int)
    if len(s)!=len(p): raise ValueError('Prediction/sample alignment mismatch')
    eligible=s[s.target!=0][['lap','onset_ms','target']].drop_duplicates().copy()
    good=(p==s.target.to_numpy())&(p!=0)
    detections=s.loc[good,['lap','onset_ms','target','input_end_ms']].copy()
    if len(detections):
        leads=detections.onset_ms-detections.input_end_ms
        assert ((leads>0)&(leads<=250)).all()
    first=detections.groupby(['lap','onset_ms','target']).input_end_ms.min().reset_index()
    eligible=eligible.merge(first,on=['lap','onset_ms','target'],how='left')
    eligible['detected']=eligible.input_end_ms.notna()
    eligible['first_warning_lead_ms']=eligible.onset_ms-eligible.input_end_ms
    episodes=[]; current=None
    for i,r in enumerate(s.itertuples()):
        if not p[i]: current=None; continue
        continuing=(current is not None and current['lap']==int(r.lap)
                    and current['prediction']==int(p[i]) and i==current['last_index']+1
                    and 0 < float(r.input_end_ms)-current['end_ms'] <=50)
        if not continuing:
            current={'lap':int(r.lap),'prediction':int(p[i]),'start_ms':float(r.input_end_ms),
                     'end_ms':float(r.input_end_ms),'windows':0,'correct_windows':0,'last_index':i}
            episodes.append(current)
        current['end_ms']=float(r.input_end_ms); current['last_index']=i
        current['windows']+=1; current['correct_windows']+=int(good[i])
    for e in episodes:
        e['false_warning']=e['correct_windows']==0; del e['last_index']
    episode_table=pd.DataFrame(episodes,columns=['lap','prediction','start_ms','end_ms','windows','correct_windows','false_warning'])
    exposure=0.; recorded=0.
    for lap in lap_ids:
        t=s.loc[s.lap==lap,'input_end_ms'].to_numpy()
        dt=np.diff(t); exposure+=float(dt[(dt>0)&(dt<=50)].sum())/60000
        times=raw.loc[raw.Lap==lap,'CurrentTime']; recorded+=float(times.max()-times.min())/60000
    all_events=confirmed_raw_onsets(raw[raw.Lap.isin(lap_ids)])
    rows=[]
    for name,c in list(LABEL_MAP.items())[1:]:
        events=eligible[eligible.target==c]
        warnings=episode_table[episode_table.prediction==c]
        false=int(warnings.false_warning.sum()) if len(warnings) else 0
        detected=int(events.detected.sum())
        lead=events.loc[events.detected,'first_warning_lead_ms']
        rows.append({'class':name,'raw_confirmed_onsets':int((all_events.target==c).sum()),
                     'eligible_onsets':len(events),'detected_onsets':detected,
                     'event_recall':detected/len(events) if len(events) else None,
                     'warning_episodes':len(warnings),'false_warning_episodes':false,
                     'false_warning_windows':int(((p==c)&~good).sum()),
                     'false_episodes_per_lap':false/len(lap_ids),
                     'false_episodes_per_recorded_minute':false/recorded if recorded else None,
                     'false_episodes_per_eligible_minute':false/exposure if exposure else None,
                     'eligible_monitoring_minutes':exposure,'recorded_minutes':recorded,
                     'mean_first_warning_lead_ms':float(lead.mean()) if len(lead) else None})
    return rows, eligible, episode_table, all_events

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',default=DEFAULT_RESULTS); args=ap.parse_args()
    out=writable_output(args.results)
    manifest=json.loads((out/'manifest.json').read_text())
    protocol=json.loads((out/'event_protocol.json').read_text())
    if protocol!=EVENT_PROTOCOL: raise ValueError('Event protocol differs from frozen settings')
    records=[]
    for domain,path in [('dry_test','data/dry_laps.csv'),('wet_test','data/wet_laps.csv')]:
        raw=read_csv(path); samples=pd.read_csv(out/f'{domain}_samples.csv')
        laps=manifest['splits']['dry_test'] if domain=='dry_test' else manifest['wet_test']['laps']
        for kind in manifest['args']['models']:
            for seed in manifest['args']['seeds']:
                predictions=pd.read_csv(out/f'{kind}_{seed}_{domain}_predictions.csv')
                assert np.array_equal(predictions.target,samples.target)
                rows,events,episodes,all_events=evaluate_events(samples,predictions.prediction,raw,laps)
                for row in rows: row.update(model=kind,seed=seed,domain=domain)
                records.extend(rows)
                events.to_csv(out/f'{kind}_{seed}_{domain}_event_detections.csv',index=False)
                episodes.to_csv(out/f'{kind}_{seed}_{domain}_warning_episodes.csv',index=False)
                all_events.to_csv(out/f'{domain}_all_confirmed_onsets.csv',index=False)
    table=pd.DataFrame(records); table.to_csv(out/'event_metrics.csv',index=False)
    cols=['event_recall','false_warning_episodes','false_episodes_per_lap',
          'false_episodes_per_recorded_minute','false_episodes_per_eligible_minute','mean_first_warning_lead_ms']
    table.groupby(['model','domain','class'])[cols].agg(['mean','std']).to_csv(out/'event_aggregate.csv')
    print(table.groupby(['model','domain','class'])[['event_recall','false_episodes_per_lap']].mean().to_string())

if __name__=='__main__': main()
