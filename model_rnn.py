import torch.nn as nn
class HandlingRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(12,64,2,batch_first=True,dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64,3)
    def forward(self,x):
        out,_ = self.rnn(x)
        return self.fc(self.dropout(out[:,-1]))
