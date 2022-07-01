import torch
import torch.nn as nn
import random 
import numpy as np

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from timm.models.layers import DropPath

from knn_cuda import KNN
from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL2

### Point Cloud Embedding Modules ###

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

class Group(nn.Module):   # FPS + KNN

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps to get the centers
        center = fps(xyz, self.num_group)
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) 
        #assert idx.size(1) == self.num_group
        #assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class Encoder(nn.Module):   ## Embedding Module

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1, bias=False), 
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1, bias=False), 
            nn.BatchNorm1d(512), 
            nn.ReLU(inplace=True), 
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        B, G, N, _ = point_groups.shape
        point_groups = point_groups.reshape(B * G, N, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1)) # B*G x 256 x N
        feature_global = torch.max(feature, dim=2, keepdim=True)[0] # B*G x 256 x 1
        # concating global features to each point features
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1) # B*G x 512 x N
        feature = self.second_conv(feature) # B*G x encoder_channel x N
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # B*G x encoder_channels
        return feature_global.reshape(B, G, self.encoder_channel)

### Transformer Modules ### (no relation with point clouds)

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k , v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super().__init__()
        
        mlp_hidden_dim = int(dim * mlp_ratio)

        # NOTE: Should test if this is better than dropout
        self.drop_path = DropPath(drop_path) if drop_path >0. else nn.Identity()

        # ATTENTION BLOCK
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        
        # MLP BLOCK
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
        for i in range(depth)])

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x

class TransformerDecoder(nn.Module):

    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):

        for block in self.blocks:
            x = block(x+pos)

        x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixel
        
        return x

class MaskedTransformer(nn.Module):

    def __init__(self,):
        super().__init__()
        
        # Network Parameters

        self.mask_ratio     = 0.6
        self.trans_dim      = 384
        self.depth          = 12
        self.drop_path_rate = 0.1 #?
        self.num_heads      = 6
        self.encoder_dims   = 384
        self.mask_type      = 'rand'

        self.build_network()

        self.apply(self._init_weights)

    def build_network(self):

        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        # dpr : drop path rate
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth, 
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )
        
        self.norm = nn.LayerNorm(self.trans_dim)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            #trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center: B x G x 3
            -----------------
            mask  : B x G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G x 3
            points = points.unsqueeze(0) # 1 x G x 3
            # selecting a random point 
            index = random.randint(0, points.size(1) - 1)
            # find the distance of this point to the other
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1) # 1x1x3 - 1xGx3 -> 1 x G
            # sort distances in ascending order --> return indexes
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0] # G
            # find number of point to mask 
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            # create the mask
            mask = torch.zeros(len(idx))
            # set masked values to 1
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device) # B x G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-num_mask),
                np.ones(num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)

    def forward(self, neighborhood, center, noaug = False):

        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)
        else: 
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)


        group_input_tokens = self.encoder(neighborhood)

        batch_size, _, F = group_input_tokens.shape

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, F)
        
        # masking pos centers
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        # positional embedding
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos

class Point_MAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.trans_dim = 384
        self.group_size = 32
        self.num_group = 64
        self.drop_path_rate = 0.1

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        # decoder 
        self.decoder_depth = 4
        self.decoder_num_heads = 6 
        

        self.build_network()

        self.loss_func = ChamferDistanceL2()

    def build_network(self):
        
        self.MAE_encoder = MaskedTransformer()

        self.decoder_pos_emb = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(), 
            nn.Linear(128, self.trans_dim))
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth, 
            drop_path_rate=dpr, 
            num_heads=self.decoder_num_heads
        )

        self.group_devider = Group(num_group=self.num_group, group_size=self.group_size)

        self.increase_dim = nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)

    def forward(self, pts, vis=False, **kwargs):

        neighborhood, center = self.group_devider(pts)

        x_vis, mask = self.MAE_encoder(neighborhood, center)

        B, _, C = x_vis.shape

        pos_emb_vis = self.decoder_pos_emb(center[~mask]).reshape(B, -1, C)
        pos_emb_mask = self.decoder_pos_emb(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emb_mask.shape

        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emb_vis, pos_emb_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape

        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)

        gt_points = neighborhood[mask].reshape(B*M, -1, 3)

        loss = self.loss_func(rebuild_points, gt_points)

        return loss

class Point_MAE_pl(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.net = Point_MAE()
        
    def forward(self, pts):

        return self.net(pts)

    def training_step(self, batch, batch_idx):

        loss = self.net(batch)

        self.log("loss", loss, on_epoch=True)

        return loss        

    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters() ,lr=0.001, weight_decay=0.05)
        sched = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=10, max_epochs=300, warmup_start_lr=1e-6, eta_min=1e-6)
        return [opt], [sched]
        
if __name__ == "__main__":
    from pytorch_lightning.loggers import WandbLogger
    from dl_lib.datasets.shapenet import ShapeNet55
    from torch.utils.data import DataLoader
    from pytorch_lightning.callbacks import LearningRateMonitor


    data_path = "/home/ioannis/Desktop/programming/data/ShapeNet55-34"
    dataset = ShapeNet55(data_path, "train")

    train_loader = DataLoader(dataset, 32, shuffle=True, num_workers=8)
    
    model = Point_MAE_pl()

    lr_monitor = LearningRateMonitor()

    logger = WandbLogger(project="Point-MAE")
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=300, logger=logger,
                        callbacks=[lr_monitor])
    
    
    trainer.fit(model, train_dataloaders=train_loader)