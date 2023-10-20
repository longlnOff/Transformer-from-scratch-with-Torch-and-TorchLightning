from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ConfigureInformation import *
from Source.models.FeedFoward import *
from Source.models.InputEmbedding import *
from Source.models.LayerNorm import *
from Source.models.MultiHeadAttention import *
from Source.models.PositionalEncoding import *
from Source.models.ResidualConnection import *


class ProjectionLayer(pl.LightningModule):
    def __init__(self, d_model, d_vocab):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_vocab)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape = [batch_size, seq_len, d_model]
        x = self.linear(x)
        # x shape = [batch_size, seq_len, d_vocab]
        x = self.softmax(x, dim=-1)
        # x shape = [batch_size, seq_len, d_vocab]
        return x