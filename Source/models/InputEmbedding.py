from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ConfigureInformation import *



class InputEmbedding(pl.LightningModule):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(self.vocab_size, self.d_model)


    def forward(self, x):
        return self.embedding(x) * torch.math.sqrt(self.d_model, dtype=torch.float32)