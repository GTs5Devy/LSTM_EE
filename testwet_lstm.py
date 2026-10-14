"""Re-evaluate frozen future-task checkpoints with their training scaler."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_lstm import read_csv,build_windows,DrivingTimeSeriesDataset
from train_lstm import metrics,predict
from experiment_config import DEFAULT_RESULTS, window_config, make_model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',default=DEFAULT_RESULTS); args=ap.parse_args()
    out=Path(args.results); scaler=np.load(out/'scaler.npz')
    manifest=json.loads((out/'manifest.json').read_text())
    d=read_csv('data/wet_laps.csv'); x,y,_=build_windows(d,sorted(d.Lap.unique()),**window_config(manifest))
    loader=DataLoader(DrivingTimeSeriesDataset(x,y,scaler['mean'],scaler['std']),batch_size=32)
    torch.set_num_threads(4)
    for kind in manifest['args']['models']:
      for seed in manifest['args']['seeds']:
        path=out/f'{kind}_{seed}.pth'
        ck=torch.load(path,weights_only=True)
        model=make_model(ck['kind'])
        model.load_state_dict(ck['state_dict']); print(path.name,json.dumps(metrics(y,predict(model,loader))))
if __name__=='__main__': main()
