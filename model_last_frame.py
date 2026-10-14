"""Optional static control: only the final past observation reaches the MLP."""
import torch.nn as nn

class LastFrameMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(12, 64), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(64, 3))

    def forward(self, x):
        return self.classifier(x[:, -1, :])
