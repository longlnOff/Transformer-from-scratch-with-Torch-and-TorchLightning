from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[0])
sys.path.append(path_git)
print('###########################################3')
print('path git: ', path_git)



import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import einops


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def get_config():
    return {
        'batch_size': 32,
        'num_epochs': 10,
        'lr': 1e-4,
        'seq_len': 359,
        'd_model': 512,
        'lang_src': 'en',
        'lang_tgt': 'it',
        'model_folder': 'model_folder',
        'model_basename': 'tmodel_',
        'preload': None,
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel',
        'train_ds_size': 0.9,
        'd_ff': 2048,
        'num_head_attention': 4,
        'num_encoder_blocks': 6,
        'num_decoder_blocks': 6,
        'dropout': 0.2,

    }


def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'

    return str(Path('.') / model_folder / model_filename)