from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ConfigureInformation import *



class MultiHeadAttention(pl.LightningModule):
    def __init__(self, d_model: int, num_head: int, dropout: float):
        super().__init__()
        assert d_model % num_head == 0
        self.d_model = d_model
        self.num_head = num_head
        self.head_dim = d_model // num_head
        self.dropout = torch.nn.Dropout(dropout)

        # create weight matrices for Q, K, V and output
        self.W_Q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_K = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_V = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_O = torch.nn.Linear(d_model, d_model, bias=False)


    @staticmethod
    def attention(Q, K, V, mask, dropout: torch.nn.Dropout):
        d_k = Q.shape[-1]

        # compute attention score
        score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # score shape = [batch_size, num_head, seq_len, seq_len]
        # apply mask
        if mask is not None:
            # replace value in score which that position is masked to -1e9
            score = score.masked_fill(mask == 0, -1e9)
        # apply softmax
        score = score.softmax(dim=-1)
        # apply dropout
        score = dropout(score)
        # compute attention
        attention = torch.matmul(score, V)
        # attention shape = [batch_size, num_head, seq_len, head_dim]

        return attention, score

    def forward(self, q, k, v, mask=None):
        # q k v shape = [batch_size, seq_len, d_model]
        # linearly transform Q, K, V
        Q = self.W_Q(q.float())
        K = self.W_K(k.float())
        V = self.W_V(v.float())
        # Q K V shape = [batch_size, seq_len, d_model]


        # split Q, K, V into num_head
        Q = einops.rearrange(Q, 'b s (h d) -> b h s d', h=self.num_head, d=self.head_dim)
        K = einops.rearrange(K, 'b s (h d) -> b h s d', h=self.num_head, d=self.head_dim)
        V = einops.rearrange(V, 'b s (h d) -> b h s d', h=self.num_head, d=self.head_dim)
        # why we need to rearrange? -> to split the d_model into num_head

        # compute attention and score
        attention, scores = MultiHeadAttention.attention(Q, K, V, mask, self.dropout)
        attention = einops.rearrange(attention, 'b h s d -> b s (h d)')
        # attention shape = [batch_size, seq_len, d_model]

        # linearly transform attention
        attention = self.W_O(attention)
        # attention shape = [batch_size, seq_len, d_model]

        # return attention, scores
        return attention
    

