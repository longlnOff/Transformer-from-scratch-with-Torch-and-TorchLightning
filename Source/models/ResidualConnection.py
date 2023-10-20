from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ConfigureInformation import *
from Source.models.LayerNorm import *

class ResidualConnection(pl.LightningModule):
    def __init__(self, dropout: float):
        super().__init__() 
        self.dropout = torch.nn.Dropout(dropout)
        self.norm    = LayerNorm()

    def forward(self, x, sublayer):
        # x shape = [batch_size, seq_len, d_model]
        # sublayer is either MultiHeadAttention or FeedForward
        # sublayer(x) shape = [batch_size, seq_len, d_model]
        # apply dropout
        x = self.dropout(sublayer(x))
        # add residual connection
        x = x + self.norm(x)
        # x shape = [batch_size, seq_len, d_model]

        return x