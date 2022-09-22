import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Callback
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import Accuracy
from dl_lib.datasets.ready_datasets import get_ModelNet40
from modules import Group, TransformerWithEmbeddings

from pipelines.finetune_pipeline import Point_MAE_finetune_pl


class PointMAEModelNet40(Point_MAE_finetune_pl):
    
    def __init__(self):
        super().__init__(40)

    def configure_networks(self):
        self.group_devider = Group(
            group_size=32, 
            num_group=64
        )

        self.MAE_encoder = TransformerWithEmbeddings(
            embed_dim=384,
            depth=1, 
            num_heads=1, 
            drop_path_rate=0.1, 
            feature_embed=True
        )

        self.cls_head = nn.Sequential(
            nn.Linear(2 * 384, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 40)
        )


    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters() ,lr=0.0005, weight_decay=0.05)
        sched = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=10, max_epochs=300, warmup_start_lr=1e-6, eta_min=1e-6)
        return [opt], [sched]



if __name__ == "__main__":

    path = '/home/ioannis/Desktop/programming/phd/PCT_Pytorch/data'
    train_loader, valid_loader = get_ModelNet40(path, 'normalized')

    model = PointMAEModelNet40()
    #model.load_submodules("/home/ioannis/Desktop/programming/phd/PointViT/custom_checkpoints/test3.pt", 
    #                      freeze_backbone=False)

    project_name = "FINETUNING POINT_MAE" 
    logger = WandbLogger(project=project_name) 

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=300, logger=logger) #, callbacks=[UnfreezeBackbone(10)])
    trainer.fit(model, train_loader, valid_loader)
