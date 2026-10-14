"""Independent boundary tests and visual sample audit."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset_lstm import *
from experiment_config import DEFAULT_RESULTS, writable_output

def self_test():
    d=pd.DataFrame({f:np.arange(80,dtype=float) for f in FEATURES})
    d['Lap']=1; d['CurrentTime']=np.arange(80)*16; d['Label']='Neutral'
    d.loc[30:32,'Label']='Understeer'
    x,y,m=build_windows(d,[1]); assert y[0]==1 and np.array_equal(x[0,:,0],np.arange(30))
    d.loc[30:32,'Label']='Neutral'; d.loc[30,'Label']='Oversteer'
    assert build_windows(d,[1])[1][0]==0
    d['Label']='Neutral'; d.loc[46:48,'Label']='Oversteer'
    assert build_windows(d,[1])[1][0]==0  # outside 250 ms
    d['Label']='Neutral'; d.loc[44:46,'Label']='Understeer'
    assert build_windows(d,[1])[1][0]==1  # onset at 240 ms, confirmation at 272 ms
    assert build_windows(d,[1],confirmation_policy='within_horizon')[1][0]==0
    d['Label']='Neutral'; d.loc[30:32,'Label']='Oversteer'; d.loc[35:37,'Label']='Understeer'
    assert build_windows(d,[1])[1][0]==2  # first confirmed onset wins
    d['Label']='Neutral'; d.loc[20,'CurrentTime']=d.loc[19,'CurrentTime']
    assert all(row['input_start_row']>=20 for row in build_windows(d,[1])[2])
    d['CurrentTime']=np.arange(80)*16; d.loc[20:,'CurrentTime']+=100
    assert all(row['input_start_row']>=20 for row in build_windows(d,[1])[2])
    d['CurrentTime']=np.arange(80)*16
    d['Lap']=np.repeat([1,2],40)
    try: build_windows(d,[1,2])
    except ValueError: pass
    else: raise AssertionError('Window crossed lap boundary')
    print('Synthetic onset, noise, horizon, persistence and lap-boundary tests passed')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',default=DEFAULT_RESULTS); args=ap.parse_args()
    self_test(); out=writable_output(args.results); out.mkdir(exist_ok=True)
    d=read_csv('data/dry_laps.csv'); x,y,m=build_windows(d,SPLITS['train'])
    fig,axes=plt.subplots(4,1,figsize=(10,12),layout='constrained')
    for target,ax in enumerate(axes):
        if target==3:
            i=next(i for i,r in enumerate(m) if r['target'] and r['confirmation_end_ms']>r['future_end_ms'])
        else:
            i=int(np.flatnonzero(y==target)[0])
        sample=m[i]
        lap=d[d.Lap==sample['lap']].sort_values('CurrentTime',kind='stable')
        z=lap.iloc[sample['input_start_row']:sample['confirmation_stop_row']]
        t=(z.CurrentTime-sample['input_end_ms'])/1000
        ax.step(t,z.Label.map(LABEL_MAP),where='post',label='Physics state')
        ax.axvspan(float(t.iloc[0]),0,color='blue',alpha=.12,label='30 past frames')
        ax.axvspan(0,.25,color='orange',alpha=.15,label='Future horizon')
        if sample['confirmation_end_ms'] is not None:
            ax.axvline((sample['onset_ms']-sample['input_end_ms'])/1000,color='green',linestyle='--',label='Onset')
            ax.axvline((sample['confirmation_end_ms']-sample['input_end_ms'])/1000,color='red',linestyle=':',label='Confirmation')
        ax.axvline(0,color='black'); ax.set_yticks([0,1,2],list(LABEL_MAP))
        ax.set_title(f"Lap {sample['lap']}: target {list(LABEL_MAP)[sample['target']]}" + (' (late onset, confirmation beyond horizon)' if target==3 else ''))
        ax.set_xlabel('Seconds relative to input end'); ax.legend(loc='upper left')
    fig.savefig(out/'sample_audit.png',dpi=160)
if __name__=='__main__': main()
