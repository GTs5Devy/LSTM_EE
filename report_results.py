"""Produce essay tables and figures from completed frozen trials."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_lstm import metrics

def main():
    out=Path('results_future'); table=pd.read_csv(out/'metrics.csv')
    paired=table.pivot(index=['model','seed'],columns='domain',values=['accuracy','macro_f1','Understeer_recall','Oversteer_recall'])
    gaps=pd.DataFrame({k:paired[k]['dry_test']-paired[k]['wet_test'] for k in ['accuracy','macro_f1','Understeer_recall','Oversteer_recall']})
    gaps.to_csv(out/'paired_dry_minus_wet.csv')
    gaps.groupby('model').agg(['mean','std']).to_csv(out/'paired_gap_summary.csv')
    baseline=[]; events=[]
    for domain in ['dry_test','wet_test']:
        s=pd.read_csv(out/(domain+'_samples.csv')); y=s.target.to_numpy(dtype=int)
        r=metrics(y,np.zeros(len(y),dtype=int)); r['domain']=domain; baseline.append(r)
        unique=s[s.target!=0].drop_duplicates(['lap','onset_ms','target'])
        events.append({'domain':domain,'windows':len(s),'understeer_onsets':int((unique.target==1).sum()),'oversteer_onsets':int((unique.target==2).sum())})
    (out/'neutral_baseline.json').write_text(json.dumps(baseline,indent=2))
    pd.DataFrame(events).to_csv(out/'distinct_onsets.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key in zip(axes,['accuracy','macro_f1']):
        for j,kind in enumerate(['lstm','rnn']):
            g=table[table.model==kind].groupby('domain')[key].agg(['mean','std']).reindex(['dry_test','wet_test'])
            ax.bar(np.arange(2)+j*.35,g['mean'],yerr=g['std'],width=.35,capsize=4,label=kind.upper())
        ax.set_xticks([.175,1.175],['Dry held-out','Wet unseen']); ax.set_ylim(0,1)
        ax.set_ylabel(key); ax.legend()
    fig.savefig(out/'model_comparison.png',dpi=160)
    summary=table.groupby(['model','domain'])[['accuracy','macro_f1','Understeer_recall','Oversteer_recall']].agg(['mean','std'])
    text='# Future-onset trial results\n\nFive seeds; error bars and standard deviations describe seed variation, not independent lap uncertainty.\n\n'
    text+='```\n'+summary.to_string()+'\n```\n\n'
    text+='Always-Neutral baselines:\n\n```\n'+pd.DataFrame(baseline)[['domain','accuracy','macro_f1']].to_string(index=False)+'\n```\n\n'
    text+='Distinct eligible onsets:\n\n```\n'+pd.DataFrame(events).to_string(index=False)+'\n```\n\n'
    text+='Targets require three consecutive event observations within 250 ms of a 30-frame, entirely Neutral input. Labels are existing physics-derived CSV labels. The forecast is conditional on a known Neutral history; live use would need an independent way to determine that eligibility. Same-width LSTM/RNN comparisons do not control parameter count. These trials remove concurrent-target leakage but cannot alone establish dependence on temporal order.\n'
    text+='\nIn these runs, both models have lower mean accuracy on wet data but higher mean macro-F1 and event recall. Thus the measured domain shift is metric-dependent and should not be described as a uniform degradation. Both models outperform the always-Neutral macro-F1 baseline, while having lower accuracy. The LSTM has higher wet mean macro-F1 than the RNN, but lower dry mean macro-F1; five seeds on fixed laps do not establish a general architectural advantage.\n'
    (out/'RESULTS.md').write_text(text)
    print(summary.to_string())
if __name__=='__main__': main()
