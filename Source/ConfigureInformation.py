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