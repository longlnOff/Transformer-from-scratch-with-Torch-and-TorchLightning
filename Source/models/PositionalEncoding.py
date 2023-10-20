from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ConfigureInformation import *


class PositionalEncoding(pl.LightningModule):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = torch.nn.Dropout(dropout)

        # craete matrix of positional encoding
        pe = torch.zeros(self.seq_len, self.d_model) # shape = [seq_len, d_model]

        # create position matrix
        position = torch.arange(0, self.seq_len, dtype=torch.float32).unsqueeze(1) # shape = [seq_len, 1]
        # we calculate positional in log scale for stability
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-torch.math.log(10000.0) / self.d_model)) # shape = [d_model/2]
        # calculate positional encoding, apply sin to even index in the array; 2i and apply cos to odd index in the array; 2i+1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension
        pe = pe.unsqueeze(0)

        # register buffer to save positional encoding (this is not model parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape = [batch_size, seq_len, d_model]
        # pe shape = [1, seq_len, d_model]
        # add positional encoding to each sequence in batch
        # below code will broadcast pe over batch dimension
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)