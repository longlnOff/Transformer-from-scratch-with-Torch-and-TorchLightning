from pathlib import Path
import sys
import os
from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from TorchLightning.ConfigureInformation import *
from TorchLightning.models.DecoderBlock import *
from TorchLightning.models.EncoderBlock import *
from TorchLightning.models.InputEmbedding import *
from TorchLightning.models.PositionalEncoding import *
from TorchLightning.models.ProjectLayer import *
from TorchLightning.DataModule import *


class Transformer(pl.LightningModule):
    def __init__(self, 
                 vocab_size_src: int, 
                 seq_len_src: int, 
                 vocab_size_tar: int, 
                 seq_len_tar: int, 
                 d_model: int,
                 d_ff: int, 
                 n_head: int,
                 epsilon: float, 
                 dropout: float,
                 num_encoder: int,
                 num_decoder: int,
                 ):
        super().__init__()
        self.vocab_size_src = vocab_size_src
        self.seq_len_src = seq_len_src
        self.vocab_size_tar = vocab_size_tar

        # init src embedding
        self.input_embedding_src = InputEmbedding(vocab_size=vocab_size_src, d_model=d_model)
        self.input_position_src  = PositionalEncoding(d_model=d_model, seq_len=seq_len_src, dropout=dropout)

        # init tar embedding
        self.input_embedding_tar = InputEmbedding(vocab_size=vocab_size_tar, d_model=d_model)
        self.input_position_tar  = PositionalEncoding(d_model=d_model, seq_len=seq_len_tar, dropout=dropout)


        # init encoder
        self.encoder = torch.nn.ModuleList([EncoderBlock(vocab_size=vocab_size_src, seq_len=seq_len_src, d_model=d_model, d_ff=d_ff, n_head=n_head, epsilon=epsilon, dropout=dropout)
                                            for i in range(num_encoder)])

        self.decoder = torch.nn.ModuleList([DecoderBlock(vocab_size=vocab_size_tar, seq_len=seq_len_tar, d_model=d_model, d_ff=d_ff, n_head=n_head, epsilon=epsilon, dropout=dropout)
                                            for i in range(num_decoder)])
        
        self.project_layer = ProjectionLayer(d_model=d_model, d_vocab=vocab_size_tar)

        self.layernorm_encoder = LayerNorm(feature_size=d_model, dropout=dropout, epsilon=epsilon)
        self.layernorm_decoder = LayerNorm(feature_size=d_model, dropout=dropout, epsilon=epsilon)


        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1, label_smoothing=0.1)

        self.loss_val = []
        
    def encode(self, src, src_mask):
        # src shape = [batch_size, seq_len]
        src = self.input_embedding_src(src)
        # src shape = [batch_size, seq_len, d_model]
        src = self.input_position_src(src)

        for layer in self.encoder:
            src = layer(src, src_mask)
        
        src = self.layernorm_encoder(src)
        # src shape = [batch_size, seq_len, d_model]

        return src
    
    def decode(self, encoder_output, src_mask, tar, tar_mask):
        tar = self.input_embedding_tar(tar)
        # tar shape = [batch_size, seq_len, d_model]
        tar = self.input_position_tar(tar)
        # tar shape = [batch_size, seq_len, d_model]
        for layer in self.decoder:
            tar = layer(tar, encoder_output, src_mask, tar_mask)
        tar = self.layernorm_decoder(tar)
        # tar shape = [batch_size, seq_len, d_model]
        return tar
    

    def project(self, x):
        return self.project_layer(x)
    
    def forward(self, encoder_input, encoder_mask, decoder_input, decoder_mask):
        encoder_output = self.encode(encoder_input, encoder_mask)
        decoder_output = self.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        projection_output = self.project(decoder_output)

        return projection_output
    
    # We can use _common_step() to reduce code duplication
    def _common_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input']
        decoder_input = batch['decoder_input']
        encoder_mask = batch['encoder_mask']
        decoder_mask = batch['decoder_mask']
        label = batch['label']
        src_text = batch['src_text']
        tgt_text = batch['tgt_text']

        projection_output = self.forward(encoder_input, encoder_mask, decoder_input, decoder_mask)
        loss = self.loss_fn(projection_output.view(-1, self.vocab_size_tar), label.view(-1))
        return loss, projection_output, label
    
    def training_step(self, batch, batch_idx):
        loss, projection_output, label = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, logger=False,)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        loss, projection_output, label = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, logger=False,)
        return {'loss': loss}

    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    

def build_model( vocab_size_src: int, 
                 vocab_size_tar: int, 
                 config: dict = None):
    seq_len_src = config['seq_len']
    seq_len_tar = config['seq_len']
    d_model = config['d_model']
    d_ff = config['d_ff']
    n_head = config['num_head_attention']
    epsilon = config['epsilon']
    dropout = config['dropout']
    num_encoder = config['num_encoder_blocks']
    num_decoder = config['num_decoder_blocks']


    
    model = Transformer(vocab_size_src=vocab_size_src,
                        seq_len_src=seq_len_src,
                        vocab_size_tar=vocab_size_tar,
                        seq_len_tar=seq_len_tar,
                        d_model=d_model,
                        d_ff=d_ff,
                        n_head=n_head,
                        epsilon=epsilon,
                        dropout=dropout,
                        num_encoder=num_encoder,
                        num_decoder=num_decoder)
    
    print(f'model size: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:02.3f}M parameters')
    
    # init weights
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    
    return model

