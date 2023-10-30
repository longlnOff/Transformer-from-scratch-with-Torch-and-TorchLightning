from pathlib import Path
import sys
import os
from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[0])
sys.path.append(path_git)
from TorchLightning.ConfigureInformation import *
from TorchLightning.models.DecoderBlock import *
from TorchLightning.models.EncoderBlock import *
from TorchLightning.models.InputEmbedding import *
from TorchLightning.models.PositionalEncoding import *
from TorchLightning.models.ProjectLayer import *
from TorchLightning.DataModule import *
from TorchLightning.models.Transformer import *





if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    config = get_config()
    model = build_model(vocab_size_src=15698,
                        vocab_size_tar=22463,
                        config=config)
                 
    


    data_module = DataModule(config=get_config())


    # Compute related
    ACCERLATOR = "gpu"
    DEVICES = [0]
    PRECISION = "16-mixed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = pl.Trainer(
        default_root_dir=config['model_folder'],
        min_epochs=1,
        max_epochs=1,
        accelerator=ACCERLATOR,
        devices=DEVICES,
        precision=PRECISION,
        callbacks=EarlyStopping(monitor='val_loss')
    )
    data_module.setup()
    trainer.fit(model=model,
                train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader())
    
    trainer.validate(model=model,
                     dataloaders=data_module.val_dataloader())