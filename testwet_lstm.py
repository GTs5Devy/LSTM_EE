"""Re-evaluate frozen future-task checkpoints with their training scaler."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_lstm import read_csv,build_windows,DrivingTimeSeriesDataset
from model_lstm import HandlingLSTM
from model_rnn import HandlingRNN
from train_lstm import metrics,predict

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',default='results_future'); args=ap.parse_args()
    out=Path(args.results); scaler=np.load(out/'scaler.npz')
    d=read_csv('data/wet_laps.csv'); x,y,_=build_windows(d,sorted(d.Lap.unique()))
    loader=DataLoader(DrivingTimeSeriesDataset(x,y,scaler['mean'],scaler['std']),batch_size=32)
    torch.set_num_threads(4)
    for path in sorted(out.glob('*.pth')):
        ck=torch.load(path,weights_only=True)
        model=HandlingLSTM() if ck['kind']=='lstm' else HandlingRNN()
        model.load_state_dict(ck['state_dict']); print(path.name,json.dumps(metrics(y,predict(model,loader))))
if __name__=='__main__': main()
