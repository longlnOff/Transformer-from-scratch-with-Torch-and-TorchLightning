from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[0])
sys.path.append(path_git)
print('###########################################')
print('path git: ', path_git)


import math
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import einops
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_config():
    return {
        'batch_size': 3,
        'num_epochs': 20,
        'lr': 1e-4,
        'seq_len': 350,
        'd_model': 512,
        'lang_src': 'en',
        'lang_tgt': 'it',
        'model_folder': f"{path_git}/Checkpoints/model_folder", 
        'model_basename': 'tmodel_',
        'preload': 'latest',
        'tokenizer_file': path_git + '/' + 'Checkpoints' + '/tokenizer_{0}.json',
        'experiment_name': f"{path_git}/Logs/runs/tmodel",
        'train_ds_size': 0.9,
        'd_ff': 2048,
        'num_head_attention': 8,
        'num_encoder_blocks': 6,
        'num_decoder_blocks': 6,
        'dropout': 0.2,

    }


def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'

    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    weights_files = list(Path(model_folder).glob(model_basename + '*'))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
