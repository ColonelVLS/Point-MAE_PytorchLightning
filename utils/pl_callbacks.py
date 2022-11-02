import pytorch_lightning as pl
from pytorch_lightning import Callback

class FreezeBackbone(Callback):

    def on_train_start(self, trainer, pl_module):
        print("Freezing Backbone")
        pl_module.freeze_backbone()

class UnfreezeBackbone(Callback):
    
    def __init__(self, unfreeze_epoch):
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.current_epoch == self.unfreeze_epoch:
            pl_module.unfreeze_backbone()
            #print("Callback Working")
