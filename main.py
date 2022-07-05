from mae_pipeline import MAESystem
from modules import Group, Mask, TransformerWithEmbeddings

import torch
import torch.nn as nn


class PointMAEPretrain(MAESystem):

    def configure_networks(self):
        self.groud_devider = Group(
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



