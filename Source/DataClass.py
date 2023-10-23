from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[0])
sys.path.append(path_git)
from Source.ConfigureInformation import *



class BilingualDataset(pl.LightningDataModule):
    def __init__(self, ds, tokenizer_src, 
                 tokenizer_tgt, src_lang, 
                 tgt_lang, seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang    
        self.seq_len = seq_len


        self.sos_token = torch.tensor(self.tokenizer_src.token_to_id("[SOS]"), dtype=torch.int64).unsqueeze(0)
        self.eos_token = torch.tensor(self.tokenizer_src.token_to_id("[EOS]"), dtype=torch.int64).unsqueeze(0)
        self.pad_token = torch.tensor(self.tokenizer_src.token_to_id("[PAD]"), dtype=torch.int64).unsqueeze(0)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]['translation']
        # print(src_target_pair)

        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        # text to ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # calculate number tokens need to padded
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 is SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # -1 is SOS


        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('sentence too long')
        
        # add sos, eos to input, after that padding
        encoder_input = torch.concat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]

        )

        decoder_input = torch.concat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        label = torch.concat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )


        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len



        return {
            'encoder_input': encoder_input,     # shape = [seq_len]
            'decoder_input': decoder_input,     # shape = [seq_len]
            'encoder_mask': einops.rearrange((encoder_input != self.pad_token).long(), 's -> 1 1 s'),        # shape = [1 1 seq_len]
            'decoder_mask': einops.rearrange((decoder_input != self.pad_token).long(), 's -> 1 1 s') & casual_mask(decoder_input.size(0)), # shape [1 1 seq_len] & [1 seq_len seq_len]
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text

        }



def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int64)
    return mask == 0