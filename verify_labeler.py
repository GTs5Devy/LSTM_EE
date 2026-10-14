"""Replay recovered functions via AST, without importing the live AC app."""
import argparse
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dataset_lstm import read_csv
from experiment_config import DEFAULT_RESULTS, writable_output

SOURCES = {'installed_v3':'labeling_source/LapTimeML_installed_v3.py',
           'github_v2':'labeling_source/LapTimeML_github_v2.py'}

def load_rule(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    size=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='history_size' for t in n.targets))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calculate_label_improved')
    scope={'history_size':size}
    for k in ['yaw','lat_accel','steer','speed']: scope[k+'_history']=deque(maxlen=size)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
    return scope

def replay(df,path):
    scope=load_rule(path); rows=[]
    # Exclude the first 20 rows of each retained lap: prior-lap state may be absent
    # after clean-lap selection, and the original app clears buffers after row 0.
    for lap,z in df.groupby('Lap',sort=False):
        for k in ['yaw','lat_accel','steer','speed']: scope[k+'_history'].clear()
        for i,r in enumerate(z.itertuples()):
            if i==0: continue
            for k,v in [('yaw',r.YawRate),('lat_accel',r.LateralAccel),('steer',r.SteerAngle),('speed',r.Speed)]:
                scope[k+'_history'].append(v)
            label,_,gradient=scope['calculate_label_improved'](r.YawRate,r.LateralAccel,r.SteerAngle,r.Speed,r.SlipDiff)
            if i>=20:
                rows.append({'lap':int(lap),'row_in_lap':i,'saved_label':r.Label,'replayed_label':label,
                             'label_match':label==r.Label,'gradient_match':bool(np.isclose(gradient,r.YawGradient,atol=1e-10,rtol=1e-8))})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',default=DEFAULT_RESULTS); args=ap.parse_args()
    out=writable_output(args.results); out.mkdir(parents=True,exist_ok=True)
    report={'source_repository':'https://github.com/GTs5Devy/Assetto_Corsa_Telemetry_Extraction',
            'github_commit':'1a76226bebad89aff838bb030d623078d9caeebf',
            'excluded_warmup_rows_per_lap':20,'sources':{},'results':[]}
    for name,path in SOURCES.items():
        report['sources'][name]={'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                 'history_size':load_rule(path)['history_size']}
        for domain in ['dry','wet']:
            d=read_csv(f'data/{domain}_laps.csv'); audit=replay(d,path)
            slip=((d.WheelSlipRL+d.WheelSlipRR)-(d.WheelSlipFL+d.WheelSlipFR))/2
            row={'source':name,'domain':domain,'audited_rows':len(audit),
                 'label_matches':int(audit.label_match.sum()),'gradient_matches':int(audit.gradient_match.sum()),
                 'slipdiff_matches':int(np.isclose(slip,d.SlipDiff,atol=1e-10,rtol=1e-8).sum()),'raw_rows':len(d)}
            report['results'].append(row)
            audit[~audit.label_match|~audit.gradient_match].to_csv(out/f'labeler_{name}_{domain}_disagreements.csv',index=False)
            if name=='installed_v3':
                assert audit.label_match.all() and audit.gradient_match.all()
                assert row['slipdiff_matches']==len(d)
            print(row)
    (out/'labeler_verification.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
