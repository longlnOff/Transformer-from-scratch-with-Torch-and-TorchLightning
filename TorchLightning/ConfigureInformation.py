from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[0])
sys.path.append(path_git)

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
from pytorch_lightning.callbacks import EarlyStopping, Callback

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def get_config():
    return {
        'batch_size': 3,
        'num_epochs': 20,
        'lr': 1e-4,
        'seq_len': 350,
        'd_model': 128,
        'lang_src': 'en',
        'lang_tgt': 'it',
        'model_folder': f"{path_git}/Checkpoints/model_folder", 
        'model_basename': 'tmodel_',
        'preload': 'latest',
        'tokenizer_file': path_git + '/' + 'Checkpoints' + '/tokenizer_{0}.json',
        'experiment_name': f"{path_git}/Logs/runs/tmodel",
        'train_ds_size': 0.9,
        'd_ff': 1024,
        'num_head_attention': 8,
        'num_encoder_blocks': 3,
        'num_decoder_blocks': 3,
        'dropout': 0.2,
        'epsilon': 1e-6,

    }