# import
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import torch


# class
class BaseTrain:
    def __init__(self, seed) -> None:
        seed_everything(seed=seed)

    def create_trainer(self, early_stopping, patience, device,
                       default_root_dir, gpus, precision, max_epochs):
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(filename='{epoch}-{step}-{val_loss:.4f}',
                            monitor='val_loss',
                            mode='min')
        ]
        if early_stopping:
            callbacks.append(
                EarlyStopping(monitor='val_loss',
                              patience=patience,
                              mode='min'))
        if device == 'cuda' and torch.cuda.is_available():
            accelerator = 'gpu'
        else:
            accelerator = 'cpu'
            gpus = 0
        # TODO: add auto_lr_find. need to read the paper to decide whether to add it.
        #       currently, the value of lr is searched by the ray tune API.
        #       the paper is here: https://arxiv.org/abs/1506.01186
        return Trainer(accelerator=accelerator,
                       callbacks=callbacks,
                       check_val_every_n_epoch=1,
                       default_root_dir=default_root_dir,
                       deterministic=True,
                       gpus=gpus,
                       precision=precision,
                       max_epochs=max_epochs)
