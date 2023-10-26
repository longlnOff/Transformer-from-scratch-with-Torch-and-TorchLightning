from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from TorchLightning.ConfigureInformation import *


class ProjectionLayer(pl.LightningModule):
    def __init__(self, d_model, d_vocab):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_vocab)

    def forward(self, x):
        # x shape = [batch_size, seq_len, d_model]
        x = self.linear(x)
        # x shape = [batch_size, seq_len, d_vocab]
        return x