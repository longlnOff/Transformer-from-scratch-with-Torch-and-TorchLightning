from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ConfigureInformation import *

class FeedFowardBlock(pl.LightningModule):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = torch.nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_2 = torch.nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x):
        # x shape = [batch_size, seq_len, d_model]
        x = self.dropout(torch.nn.functional.relu(self.linear_1(x)))
        # x shape = [batch_size, seq_len, d_ff]
        x = self.linear_2(x)
        # x shape = [batch_size, seq_len, d_model]
        return x