import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from modules import Group, TransformerWithEmbeddings
from pipelines.finetune_pipeline import Point_MAE_finetune_pl
from dl_lib.datasets import get_scanObjectNN


class PointMAEScanObjectNN(Point_MAE_finetune_pl):
    
    def __init__(self):
        super().__init__(15) # classes for ScanObjectNN

    def configure_networks(self):
        self.group_devider = Group(
            group_size=32, 
            num_group=128
        )

        self.MAE_encoder = TransformerWithEmbeddings(
            embed_dim=384,
            depth=1, 
            num_heads=6, 
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
            nn.Linear(256, self.num_classes)
        )

   
    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters() ,lr=0.0005, weight_decay=0.05)
        sched = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=10, max_epochs=300, warmup_start_lr=1e-6, eta_min=1e-6)
        return [opt], [sched]



if __name__ == "__main__":

    path = "/home/ioannis/Desktop/programming/data/ScanObjectNN/main_split/"
    train_loader, valid_loader = get_scanObjectNN(path)

    model = PointMAEScanObjectNN()

    project_name = "FINETUNING POINT_MAE - ScanOnjectNN" 
    logger = WandbLogger(project=project_name) 

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=300, logger=logger) #, callbacks=[UnfreezeBackbone(100)])
    trainer.fit(model, train_loader, valid_loader)

