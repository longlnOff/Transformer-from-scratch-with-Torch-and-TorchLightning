from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[2])
sys.path.append(path_git)
from TorchLightning.ConfigureInformation import *


class LayerNorm(pl.LightningModule):
    def __init__(self, feature_size: int, dropout: float, epsilon: float = 1e-7):
        super().__init__()
        self.gammar  = torch.nn.Parameter(torch.ones(feature_size))   # multiplicative
        self.beta    = torch.nn.Parameter(torch.zeros(feature_size))  # additive
        self.dropout = torch.nn.Dropout(dropout)
        self.epsilon = epsilon


    def forward(self, x):
        # x shape = [batch_size, seq_len, d_model]
        mean = torch.mean(input=x, dim=-1, keepdim=True)
        std  = torch.std(input=x, dim=-1, keepdim=True)
        x    = (x - mean) / torch.math.sqrt(std + self.epsilon) 
        x    = self.dropout(x*self.gammar + self.beta)
        return x
    

class ResidualConnection(pl.LightningModule):
    def __init__(self, feature_size: int, dropout: float, epsilon: float = 1e-7):
        super().__init__()
        self.layernorm = LayerNorm(feature_size=feature_size, dropout=dropout, epsilon=epsilon)
        self.dropout   = torch.nn.Dropout(dropout)

    def forward(self, x, transformed_input):
        x = x + self.dropout(self.layernorm(transformed_input))
        return x
    

if __name__ == '__main__':
    layernorm = LayerNorm(feature_size=512, dropout=0.1)
    print(f'model size: {sum(p.numel() for p in layernorm.parameters() if p.requires_grad) / 1e6:02.3f}M parameters')
