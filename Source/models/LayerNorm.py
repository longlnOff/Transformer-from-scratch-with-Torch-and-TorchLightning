from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ConfigureInformation import *



class LayerNorm(pl.LightningModule):
    def __init__(self, features: int = 512, epsilon: float = 10e-6, dropout: float = 0.1):
        super().__init__()
        self.epsilon = epsilon  # epsilon is a small value to avoid division by zero
        self.gamma = torch.nn.Parameter(torch.ones(features))  # multiply by gamma
        self.beta = torch.nn.Parameter(torch.zeros(features))  # add beta
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x shape = [batch_size, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.dropout(self.gamma * (x - mean) / (std + self.epsilon) + self.beta)