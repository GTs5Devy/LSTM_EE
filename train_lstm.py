"""Reproducible dry-only training and frozen dry/wet evaluation."""
import argparse, copy, hashlib, json, random, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset_lstm import FEATURES, LABEL_MAP, SPLITS, read_csv, build_windows, DrivingTimeSeriesDataset
from model_lstm import HandlingLSTM
from model_rnn import HandlingRNN

def metrics(y,p):
    cm = np.zeros((3,3),dtype=int)
    np.add.at(cm,(y,p),1)
    precision = np.divide(cm.diagonal(),cm.sum(0),out=np.zeros(3),where=cm.sum(0)>0)
    recall = np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(3),where=cm.sum(1)>0)
    f1 = np.divide(2*precision*recall,precision+recall,out=np.zeros(3),where=precision+recall>0)
    r = {'accuracy':float(np.mean(y==p)), 'macro_f1':float(f1.mean()), 'confusion_matrix':cm.tolist()}
    for i,name in enumerate(LABEL_MAP):
        for key,val in [('precision',precision[i]),('recall',recall[i]),('f1',f1[i])]: r[name+'_'+key]=float(val)
    return r

def predict(model,loader):
    model.eval(); result=[]
    with torch.no_grad():
        for x,_ in loader: result.extend(model(x).argmax(1).numpy().tolist())
    return np.array(result)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    ap.add_argument('--models',nargs='+',choices=['lstm','rnn'],default=['lstm','rnn'])
    ap.add_argument('--out',default='results_future'); ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args(); out=Path(args.out); out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    dry=read_csv('data/dry_laps.csv')
    assert set(dry.Lap.unique()) == set(sum(SPLITS.values(),[]))
    assert len(sum(SPLITS.values(),[])) == len(set(sum(SPLITS.values(),[])))
    raw={k:build_windows(dry,v) for k,v in SPLITS.items()}
    mean=raw['train'][0].mean(axis=(0,1),dtype=np.float64)
    std=raw['train'][0].std(axis=(0,1),dtype=np.float64); std[std<1e-8]=1
    np.savez(out/'scaler.npz',mean=mean,std=std)
    counts=np.bincount(raw['train'][1],minlength=3)
    if np.any(counts==0): raise ValueError('Training split lacks a class')
    weights=counts.sum()/(3*counts)
    manifest={'splits':SPLITS,'features':FEATURES,'class_order':list(LABEL_MAP),'horizon_ms':250,'input_frames':30,'persistence_frames':3,'max_gap_ms':50,'scaler_source':'dry training windows only','class_weights':weights.tolist(),'args':vars(args),'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__},'sha256':{}}
    for path in Path('data').glob('*.csv'): manifest['sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for k,(x,y,m) in raw.items():
        c=np.bincount(y,minlength=3); print(k,'laps',SPLITS[k],'counts',c.tolist(),flush=True)
        manifest[k]={'counts':c.tolist(),'percentages':(100*c/len(y)).tolist()}
        pd.DataFrame(m).to_csv(out/(k+'_samples.csv'),index=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if args.prepare_only: return
    ds={k:DrivingTimeSeriesDataset(x,y,mean,std) for k,(x,y,_) in raw.items()}
    loaders={k:DataLoader(d,batch_size=32,shuffle=False) for k,d in ds.items()}
    all_results=[]
    # Finish and freeze every dry-selected checkpoint before loading wet telemetry.
    for kind in args.models:
        for seed in args.seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model=HandlingLSTM() if kind=='lstm' else HandlingRNN()
            optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
            criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
            train=DataLoader(ds['train'],batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
            best=float('inf'); history=[]; best_epoch=0
            for epoch in range(1,args.epochs+1):
                model.train(); total=0; denom=0
                for x,y in train:
                    optimizer.zero_grad(); loss=criterion(model(x),y); loss.backward(); optimizer.step()
                    w=weights[y.numpy()].sum(); total+=loss.item()*w; denom+=w
                model.eval(); vt=0; vd=0
                with torch.no_grad():
                    for x,y in loaders['validation']:
                        w=weights[y.numpy()].sum(); vt+=criterion(model(x),y).item()*w; vd+=w
                val=vt/vd; history.append({'epoch':epoch,'train_loss':total/denom,'validation_loss':val})
                if val<best:
                    best=val; best_epoch=epoch; state=copy.deepcopy(model.state_dict())
                print(kind,seed,epoch,'val loss',round(val,5),flush=True)
            torch.save({'state_dict':state,'kind':kind,'seed':seed,'best_epoch':best_epoch,'parameters':sum(p.numel() for p in model.parameters())},out/f'{kind}_{seed}.pth')
            pd.DataFrame(history).to_csv(out/f'{kind}_{seed}_history.csv',index=False)
    wet=read_csv('data/wet_laps.csv'); wx,wy,wm=build_windows(wet,sorted(wet.Lap.unique()))
    pd.DataFrame(wm).to_csv(out/'wet_test_samples.csv',index=False)
    loaders['wet_test']=DataLoader(DrivingTimeSeriesDataset(wx,wy,mean,std),batch_size=32)
    manifest['wet_test']={'laps':sorted(int(v) for v in wet.Lap.unique()),'counts':np.bincount(wy,minlength=3).tolist()}
    for kind in args.models:
        for seed in args.seeds:
            ck=torch.load(out/f'{kind}_{seed}.pth',weights_only=True)
            model=HandlingLSTM() if kind=='lstm' else HandlingRNN(); model.load_state_dict(ck['state_dict'])
            for domain in ['dry_test','wet_test']:
                y=raw[domain][1] if domain=='dry_test' else wy
                p=predict(model,loaders[domain]); r=metrics(y,p)
                r.update(model=kind,seed=seed,domain=domain,best_epoch=ck['best_epoch'],parameters=ck['parameters'])
                all_results.append(r)
                pd.DataFrame({'target':y,'prediction':p}).to_csv(out/f'{kind}_{seed}_{domain}_predictions.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(all_results,indent=2))
    table=pd.DataFrame([{k:v for k,v in r.items() if k!='confusion_matrix'} for r in all_results]); table.to_csv(out/'metrics.csv',index=False)
    cols=[c for c in table if c not in ['model','seed','domain','best_epoch','parameters']]
    table.groupby(['model','domain'])[cols].agg(['mean','std']).to_csv(out/'aggregate.csv')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(table[['model','seed','domain','accuracy','macro_f1']].to_string(index=False),flush=True)
if __name__=='__main__': main()
