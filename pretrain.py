from mae_pipeline import MAESystem
from modules import Group, Mask, TransformerWithEmbeddings

import torch
import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

class PointMAEPretrain(MAESystem):

    def configure_networks(self):
        self.group_devider = Group(
            group_size=32, 
            num_group=64
        )

        self.mask_generator = Mask(
            mask_ratio=0.6, 
            mask_type='rand'
        )

        self.MAE_encoder = TransformerWithEmbeddings(
            embed_dim=384,
            depth=12, 
            num_heads=6, 
            drop_path_rate=0.1, 
            feature_embed=True
        )

        self.MAE_decoder = TransformerWithEmbeddings(
            embed_dim=384, 
            depth=4, 
            num_heads=6,
            drop_path_rate=0.1, 
            feature_embed=False
        )

        # 384 = embed_dim , 32 = group_size
        self.increase_dim = nn.Conv1d(384, 3 * 32, 1)


    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters() ,lr=0.001, weight_decay=0.05)
        sched = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=10, max_epochs=300, warmup_start_lr=1e-6, eta_min=1e-6)
        return [opt], [sched]


if __name__=="__main__":

    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import LearningRateMonitor
    from dl_lib.datasets.shapenet import ShapeNet55
    from torch.utils.data import DataLoader

    data_path = "/home/ioannis/Desktop/programming/data/ShapeNet55-34"
    dataset = ShapeNet55(data_path, 'train')

    train_loader = DataLoader(dataset, 32, shuffle=True, num_workers=8)
    
    model = PointMAEPretrain()

    lr_monitor = LearningRateMonitor()

    logger = WandbLogger(project="Point-MAE")
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=5, logger=logger,
                        callbacks=[lr_monitor])
        
    trainer.fit(model, train_dataloaders=train_loader)

    # save custom checkpoint
    checkpoint_path = "/home/ioannis/Desktop/programming/phd/PointViT/custom_checkpoints/test_ckpt.pt"
    model.save_submodules(checkpoint_path)


    