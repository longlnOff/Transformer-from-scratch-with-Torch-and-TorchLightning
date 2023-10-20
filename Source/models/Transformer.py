from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ConfigureInformation import *
from Source.models.FeedFoward import *
from Source.models.MultiHeadAttention import *
from Source.models.PositionalEncoding import *
from Source.models.Encoder import *
from Source.models.Decoder import *
from Source.models.InputEmbedding import *
from Source.models.LayerNorm import *
from Source.models.ResidualConnection import *
from Source.models.ProjectionLayer import *


class Transformer(pl.LightningModule):
    def __init__(self,
                 encoder: StackEncoder,
                 decoder: StackDecoder,
                 src_embedding: InputEmbedding,
                 tar_embedding: InputEmbedding,
                 src_position: PositionalEncoding,
                 tar_position: PositionalEncoding,
                 projection: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tar_embedding = tar_embedding
        self.src_position = src_position
        self.tar_position = tar_position
        self.projection = projection


    def encode(self, src, src_mask):
        # src shape = [batch_size, seq_len]
        # src_mask shape = [batch_size, seq_len]
        src = self.src_embedding(src)
        # src shape = [batch_size, seq_len, d_model]
        src = self.src_position(src)
        # src shape = [batch_size, seq_len, d_model]
        src = self.encoder(src, src_mask)
        # src shape = [batch_size, seq_len, d_model]
        return src
    
    def decode(self, encoder_output, src_mask, tar, tar_mask):
        tar = self.tar_embedding(tar)
        # tar shape = [batch_size, seq_len, d_model]
        tar = self.tar_position(tar)
        # tar shape = [batch_size, seq_len, d_model]
        tar = self.decoder(tar, encoder_output, src_mask, tar_mask)
        # tar shape = [batch_size, seq_len, d_model]
        return tar
    
    def project(self, x):
        return self.projection(x)
    


def build_model(src_vocab_size: int, 
                tar_vocab_size: int, 
                src_seq_len: int, 
                tar_seq_len: int,
                d_model: int,
                d_ff: int,
                num_head_attention: int,
                num_encoder_blocks: int,
                num_decoder_blocks: int,
                num_feed_forward: int,
                dropout: float,
                device: str):
    
    src_embedding = InputEmbedding(src_vocab_size, d_model)
    tar_embedding = InputEmbedding(tar_vocab_size, d_model)
    src_position = PositionalEncoding(d_model, dropout, src_seq_len)
    tar_position = PositionalEncoding(d_model, dropout, tar_seq_len)
    

    # create encoder
    encoder_bocks = []
    for _ in range(num_encoder_blocks):
        encoder_self_attention = MultiHeadAttention(d_model, num_head_attention, dropout)
        feed_forward_block = FeedFowardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, feed_forward_block, dropout)
        encoder_bocks.append(encoder_block)

    # create decoder
    decoder_blocks = []
    for _ in range(num_decoder_blocks):
        decoder_self_attention = MultiHeadAttention(d_model, num_head_attention, dropout)
        decoder_encoder_attention = MultiHeadAttention(d_model, num_head_attention, dropout)
        feed_forward_block = FeedFowardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_encoder_attention, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # projection layer
    projection = ProjectionLayer(d_model, tar_vocab_size)

    # create encoder and decoder
    encoder = StackEncoder(torch.nn.ModuleList(encoder_bocks))
    decoder = StackDecoder(torch.nn.ModuleList(decoder_blocks))

    # create transformer
    transformer = Transformer(encoder, decoder, src_embedding, tar_embedding, src_position, tar_position, projection)
