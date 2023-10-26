from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from TorchLightning.ConfigureInformation import *
from TorchLightning.models.modules.AddAndNorm import *
from TorchLightning.models.modules.FeedForward import *
from TorchLightning.models.modules.MultiHeadAttention import *


class DecoderBlock(pl.LightningModule):
    def __init__(self, 
                 vocab_size: int, 
                 seq_len: int, 
                 d_model: int,
                 d_ff: int, 
                 n_head: int,
                 epsilon: float, 
                 dropout: float):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model=d_model,
                                      n_head=n_head,
                                      dropout=dropout)
        
        self.mha2 = MultiHeadAttention(d_model=d_model,
                                      n_head=n_head,
                                      dropout=dropout)
        
        self.add_and_norm1 = ResidualConnection(feature_size=d_model,
                                               dropout=dropout,
                                               epsilon=epsilon)
        
        self.add_and_norm2 = ResidualConnection(feature_size=d_model,
                                               dropout=dropout,
                                               epsilon=epsilon)
        
        self.add_and_norm3 = ResidualConnection(feature_size=d_model,
                                               dropout=dropout,
                                               epsilon=epsilon)
        
        self.feed_forward = FeedForward(d_model=d_model,
                                        d_ff=d_ff,
                                        dropout=dropout)
        
    def forward(self, x, input_from_encoder, src_mask, tar_mask):
        
        y = self.mha1(q=x, k=x, v=x, mask=tar_mask)
        x = self.add_and_norm1(x=x, transformed_input=y)

        y = self.mha2(q=x, k=input_from_encoder, v=input_from_encoder, mask=src_mask)
        x = self.add_and_norm2(x=x, transformed_input=y)

        y = self.feed_forward(x)
        x = self.add_and_norm3(x=x, transformed_input=y)

        return x

