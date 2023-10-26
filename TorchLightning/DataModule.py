from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[0])
sys.path.append(path_git)
from TorchLightning.ConfigureInformation import *



class BilinguaDataset(pl.LightningDataModule):
    def __init__(self):

        pass