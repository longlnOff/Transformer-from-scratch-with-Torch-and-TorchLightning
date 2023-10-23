from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ConfigureInformation import *
from Source.models.FeedForward import *
from Source.models.InputEmbedding import *
from Source.models.LayerNorm import *
from Source.models.MultiHeadAttention import *
from Source.models.PositionalEncoding import *
from Source.models.ResidualConnection import *



class EncoderBlock(pl.LightningModule):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_1 = ResidualConnection(dropout)
        self.residual_connection_2 = ResidualConnection(dropout)


    def forward(self, x, mask=None):
        # x shape = [batch_size, seq_len, d_model]
        # self_attention_block(x) shape = [batch_size, seq_len, d_model]
        # feed_forward_block(x) shape = [batch_size, seq_len, d_model]
        # apply residual connection
        x = self.residual_connection_1(x, lambda x: self.self_attention_block(x, x, x, mask))
        # apply residual connection
        x = self.residual_connection_2(x, self.feed_forward_block)
        # x shape = [batch_size, seq_len, d_model]

        return x
    

class StackEncoder(pl.LightningModule):
    def __init__(self, layers: torch.nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask=None):
        # x shape = [batch_size, seq_len, d_model]
        for layer in self.layers:
            x = layer(x, mask)
        # x shape = [batch_size, seq_len, d_model]
        x = self.norm(x)
        # x shape = [batch_size, seq_len, d_model]
        return x