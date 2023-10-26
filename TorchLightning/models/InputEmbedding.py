from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from TorchLightning.ConfigureInformation import *



class InputEmbedding(pl.LightningModule):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, input):
        # input shape = [B, seq_len]
        output = self.embedding(input)
        # output shape = [B, seq_len, d_model]
        return output * math.sqrt(self.d_model)