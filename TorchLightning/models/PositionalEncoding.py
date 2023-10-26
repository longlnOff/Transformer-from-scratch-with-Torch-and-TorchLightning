from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from TorchLightning.ConfigureInformation import *

class PositionalEncoding(pl.LightningModule):
    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, self.seq_len, dtype=int)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))      # small modify

        pe[pos, 0::2] = torch.sin(pos.unsqueeze(1).float()@div_term.unsqueeze(0))
        pe[pos, 1::2] = torch.cos(pos.unsqueeze(1).float()@div_term.unsqueeze(0))

        # add batch dimension
        pe = pe.unsqueeze(0)        # [1, seq_len, d_model]

        # register buffer to save positional encoding (this is not model parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape = [batch_size, seq_len, d_model]
        # add positional encoding to each sequence in batch
        # below code will broadcast pe over batch dimension
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)