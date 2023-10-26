from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[2])
sys.path.append(path_git)
from TorchLightning.ConfigureInformation import *


class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff    = d_ff
        self.dropout = torch.nn.Dropout(dropout)

        self.ffw1    = torch.nn.Linear(d_model, d_ff, bias=True)
        self.ffw2    = torch.nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        # x shape = [B, seq_len, d_model]
        x = self.dropout(torch.nn.functional.relu(self.ffw1(x)))
        x = self.ffw2(x)

        return x