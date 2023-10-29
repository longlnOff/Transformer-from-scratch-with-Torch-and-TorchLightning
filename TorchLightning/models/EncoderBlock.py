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



class EncoderBlock(pl.LightningModule):
    def __init__(self, 
                 vocab_size: int, 
                 seq_len: int, 
                 d_model: int,
                 d_ff: int, 
                 n_head: int,
                 epsilon: float, 
                 dropout: float):
        super().__init__()

        self.mha = MultiHeadAttention(d_model=d_model,
                                      n_head=n_head,
                                      dropout=dropout)
        
        self.add_and_norm1 = ResidualConnection(feature_size=d_model,
                                               dropout=dropout,
                                               epsilon=epsilon)
        
        self.add_and_norm2 = ResidualConnection(feature_size=d_model,
                                               dropout=dropout,
                                               epsilon=epsilon)
        
        self.feed_forward = FeedForward(d_model=d_model,
                                        d_ff=d_ff,
                                        dropout=dropout)
        

    def forward(self, x, mask):
        # mha is combined with add&norm
        out_layer, scores = self.mha(q=x, k=x, v=x, mask=mask)
        x = self.add_and_norm1(x=x, transformed_input=out_layer)

        # ffw is combined with add&norm
        out_layer = self.feed_forward(x)
        x = self.add_and_norm2(x=x, transformed_input=out_layer)

        return x