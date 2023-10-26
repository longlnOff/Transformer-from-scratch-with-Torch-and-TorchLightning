from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from TorchLightning.ConfigureInformation import *
from TorchLightning.models.DecoderBlock import *
from TorchLightning.models.EncoderBlock import *
from TorchLightning.models.InputEmbedding import *
from TorchLightning.models.PositionalEncoding import *
from TorchLightning.models.ProjectLayer import *



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
    

def build_model( vocab_size_src: int, 
                 seq_len_src: int, 
                 vocab_size_tar: int, 
                 seq_len_tar: int, 
                 d_model: int,
                 d_ff: int, 
                 n_head: int,
                 epsilon: float, 
                 dropout: float,
                 num_encoder: int,
                 num_decoder: int,):
    
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
    
    # init weights
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    
    return model

if __name__ == "__main__":
    model = build_model(vocab_size_src=15698,
                        seq_len_src=350,
                        vocab_size_tar=22463,
                        seq_len_tar=274,
                        d_model=512,
                        d_ff=2048,
                        n_head=8,
                        epsilon=1e-6,
                        dropout=0.1,
                        num_encoder=6,
                        num_decoder=6)
    
    print(f'model size: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:02.3f}M parameters')
