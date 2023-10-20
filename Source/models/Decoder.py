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


class DecoderBlock(pl.LightningModule):
    def __init__(self, self_attention_block: MultiHeadAttention,
                       cross_attention_block: MultiHeadAttention,
                       feed_forward_block: FeedFowardBlock,
                       dropout: float):
        super().__init__()
        self.dropout = dropout
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_1 = ResidualConnection(self.dropout)
        self.residual_connection_2 = ResidualConnection(self.dropout)
        self.residual_connection_3 = ResidualConnection(self.dropout)


    def forward(self, y, encoder_output, src_mask, target_mask):
                                         # encoder mask, decoder mask
        # y shape = [batch_size, seq_len, d_model]
        # self attention for decoder
        y = self.residual_connection_1(y, lambda y: self.self_attention_block(y, y, y, target_mask))
        # y shape = [batch_size, seq_len, d_model]
        # cross attention for decoder with encoder output, Q froms decoder, K and V from encoder
        y = self.residual_connection_2(y, lambda y: self.cross_attention_block(y, encoder_output, encoder_output, src_mask))
        # y shape = [batch_size, seq_len, d_model]
        # feed forward
        y = self.residual_connection_3(y, self.feed_forward_block)
        # y shape = [batch_size, seq_len, d_model]
        return y
    

class StackDecoder(pl.LightningModule):
    def __init__(self, layers: torch.nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        x = self.norm(x)
        return x