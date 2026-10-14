"""Produce essay tables and figures from completed frozen trials."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_lstm import metrics
from experiment_config import DEFAULT_RESULTS, writable_output

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',default=DEFAULT_RESULTS); args=ap.parse_args()
    out=writable_output(args.results); table=pd.read_csv(out/'metrics.csv')
    models=table.model.drop_duplicates().tolist()
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
    width=.8/len(models)
    for ax,key in zip(axes,['macro_f1','accuracy']):
        for j,kind in enumerate(models):
            g=table[table.model==kind].groupby('domain')[key].agg(['mean','std']).reindex(['dry_test','wet_test'])
            ax.bar(np.arange(2)+j*width,g['mean'],yerr=g['std'],width=width,capsize=4,label=kind.upper())
        ax.set_xticks(np.arange(2)+(len(models)-1)*width/2,['Dry held-out','Wet unseen']); ax.set_ylim(0,1)
        ax.set_ylabel(key); ax.set_title('Primary metric' if key=='macro_f1' else 'Supplementary metric'); ax.legend()
    fig.savefig(out/'model_comparison.png',dpi=160)
    summary=table.groupby(['model','domain'])[['macro_f1','Understeer_precision','Understeer_recall','Oversteer_precision','Oversteer_recall','accuracy']].agg(['mean','std'])
    text='# Future-onset trial results\n\nFive seeds; error bars and standard deviations describe seed variation, not independent lap uncertainty.\n\n'
    text+='```\n'+summary.to_string()+'\n```\n\n'
    text+='Always-Neutral baselines:\n\n```\n'+pd.DataFrame(baseline)[['domain','accuracy','macro_f1']].to_string(index=False)+'\n```\n\n'
    text+='Distinct eligible onsets:\n\n```\n'+pd.DataFrame(events).to_string(index=False)+'\n```\n\n'
    text+='An onset must start within 250 ms after 30 consecutive Neutral telemetry observations (~0.5 s). The following two matching observations may extend beyond 250 ms solely for persistence confirmation. Labels are recovered logger-derived CSV labels. Macro-F1 and class precision/recall are primary; accuracy is supplementary. Forecasts remain conditional on Neutral-history eligibility. Same-width LSTM/RNN comparisons do not control parameter count.\n'
    for kind in models:
        means=table[table.model==kind].groupby('domain')[['macro_f1','accuracy']].mean()
        text+=f"\n{kind}: dry macro-F1 {means.loc['dry_test','macro_f1']:.3f}, wet {means.loc['wet_test','macro_f1']:.3f}; dry accuracy {means.loc['dry_test','accuracy']:.3f}, wet {means.loc['wet_test','accuracy']:.3f}.\n"
    if 'last_frame' in models:
        p=table.pivot(index='seed',columns=['model','domain'],values='macro_f1'); contrasts=[]
        for kind in ['lstm','rnn']:
            if kind in models:
                for domain in ['dry_test','wet_test']:
                    difference=p[(kind,domain)]-p[('last_frame',domain)]
                    contrasts.extend({'model':kind,'domain':domain,'seed':int(seed),'macro_f1_minus_last_frame':float(value)} for seed,value in difference.items())
        c=pd.DataFrame(contrasts); c.to_csv(out/'sequence_minus_last_frame.csv',index=False)
        cs=c.groupby(['model','domain']).macro_f1_minus_last_frame.agg(['mean','std'])
        text+='\nOptional last-frame control: paired sequence-minus-last-frame macro-F1:\n\n```\n'+cs.to_string()+'\n```\n\n'
        text+='This control tests the value of these sequence models relative to this static MLP. Different architectures/capacities prevent a causal memory-only conclusion.\n'
    if (out/'event_metrics.csv').exists():
        e=pd.read_csv(out/'event_metrics.csv')
        es=e.groupby(['model','domain','class'])[['event_recall','false_episodes_per_lap','false_episodes_per_eligible_minute']].agg(['mean','std'])
        text+='\nOptional event-level evaluation:\n\n```\n'+es.to_string()+'\n```\n\n'
        text+='Event recall counts distinct eligible first-confirmed onsets, not all raw events. False-warning episodes merge consecutive same-class warnings at <=50 ms gaps; an episode is false if it contains no correct target warning. Rates are conditional on eligible monitoring. See event_protocol.json and raw/eligible event counts.\n'
    text='\n'.join(line.rstrip() for line in text.splitlines()).rstrip()+'\n'
    (out/'RESULTS.md').write_text(text,encoding='utf-8')
    print(summary.to_string())
if __name__=='__main__': main()
