from pathlib import Path
import sys
import os

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[0])
sys.path.append(path_git)
from TorchLightning.ConfigureInformation import *



class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config


    def get_all_sentences(self, ds, lang):
        for item in ds:
            yield item['translation'][lang]

    def get_or_build_tokenizer(self, config, ds, lang):
        tokenizer_path = Path(config['tokenizer_file'].format(lang))
        if not Path.exists(tokenizer_path):
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
                min_frequency=2,)
            tokenizer.train_from_iterator(self.get_all_sentences(ds, lang), trainer)
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))

        return tokenizer
    def prepare_data(self) -> None:
        # download dataset
        ds_raw = load_dataset('opus_books', 
                        f'{self.config["lang_src"]}-{self.config["lang_tgt"]}',
                        split='train', cache_dir=path_git + '/' + 'DataFolder')
    
    def setup(self, stage: str = '') -> None:
        ds_raw = load_dataset('opus_books', 
                        f'{self.config["lang_src"]}-{self.config["lang_tgt"]}',
                        split='train', cache_dir=path_git + '/' + 'DataFolder')
    
        # Build tokenizer
        tokenizer_src = self.get_or_build_tokenizer(self.config, ds_raw, self.config['lang_src'])
        tokenizer_tgt = self.get_or_build_tokenizer(self.config, ds_raw, self.config['lang_tgt'])

        # Keep 90% for training, 10% for validation
        torch.manual_seed(0)
        train_ds_size = int(self.config['train_ds_size'] * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
        train_ds_raw = val_ds_raw

        self.train_ds = BilingualDataset(train_ds_raw,
                            tokenizer_src,
                            tokenizer_tgt,
                            self.config['lang_src'],
                            self.config['lang_tgt'],
                            self.config['seq_len'],)

        self.val_ds = BilingualDataset(val_ds_raw,
                                    tokenizer_src,
                                    tokenizer_tgt,
                                    self.config['lang_src'],
                                    self.config['lang_tgt'],
                                    self.config['seq_len'],)
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataloader = DataLoader(self.train_ds,
                                    batch_size=self.config['batch_size'],
                                    num_workers=4,
                                    shuffle=True)
        
        return train_dataloader
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataloader = DataLoader(self.val_ds,
                                    batch_size=1,
                                    num_workers=4,
                                    shuffle=False)
        return val_dataloader









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



if __name__ == '__main__':
    data_module = DataModule(config=get_config())
    data_module.setup()
    for i in data_module.train_dataloader():
        print(i)
        break