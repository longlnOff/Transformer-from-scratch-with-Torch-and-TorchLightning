from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[2])
sys.path.append(path_git)
from TorchLightning.ConfigureInformation import *




class MultiHeadAttention(pl.LightningModule):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0, 'n_head must be divisible by d_model'
        self.d_model = d_model
        self.n_head  = n_head
        self.d_head  = int(self.d_model / self.n_head)
        self.dropout = torch.nn.Dropout(dropout)

        self.w_q     = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_k     = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_v     = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_o     = torch.nn.Linear(d_model, d_model, bias=False)


    @staticmethod
    def attention_calculate(self, q, k, v, mask, dropout: torch.nn.Dropout):
        # shape q, k, v = [B, s, d_k]
        # mask shape = [B, s]
        d_k = q.shape[-1]
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            score = score.masked_fill(mask=(mask == 0), value=1e-10)
        # score shape = [B, s]

        score = torch.nn.functional.softmax(score, dim=-1)
        score = dropout(score)

        attention = torch.matmul(score, v)
        # shape attention = [B, s, d_k]
        return attention, score
    

    def forward(self, q, k, v, mask):
        q = self.w_q(q).float() # shape = [B, s, d_model]
        k = self.w_k(k).float() # shape = [B, s, d_model]
        v = self.w_v(v).float() # shape = [B, s, d_model]

        # convert to multi-head
        q = einops.rearrange(q, 'b s (h d) -> b h s d', h=self.n_head, d=self.d_head)
        # shape = [B, H, S, d_head]
        k = einops.rearrange(k, 'b s (h d) -> b h s d', h=self.n_head, d=self.d_head)
        # shape = [B, H, S, d_head]
        v = einops.rearrange(v, 'b s (h d) -> b h s d', h=self.n_head, d=self.d_head)
        # shape = [B, H, S, d_head]

        attention, score = MultiHeadAttention.attention_calculate(q=q, k=k, v=v,
                                                                  mask=mask, dropout=self.dropout)
        
        attention = einops.rearrange(attention, 'b h s d -> b s (h d)', h=self.n_head, d=self.d_head)
        # attention shape = [B, s, d_model]

        # Linear transform
        attention = self.w_o(attention)

        self.attention_scores = score

        return attention, score