import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy
from dl_lib.datasets.ready_datasets import get_ModelNet40

from modules import Group, TransformerWithEmbeddings

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

###############################
#    Classification System    #
###############################

class Point_MAE_finetune_pl(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
    
        self.configure_networks()

    def configure_networks(self):
        self.group_devider = Group(
            group_size=32, 
            num_group=64
        )

        self.MAE_encoder = TransformerWithEmbeddings(
            embed_dim=384,
            depth=12, 
            num_heads=6, 
            drop_path_rate=0.1, 
            feature_embed=True
        )

        self.cls_head = nn.Sequential(
            nn.Linear(384, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 40)
        )

    def forward(self, pts):
        neighborhood, center = self.group_devider(pts)
        x_vis = self.MAE_encoder(neighborhood, center)
        feature_vector = torch.max(x_vis, dim=1).values
        logits = self.cls_head(feature_vector)
        return logits

    def training_step(self, batch, batch_idx):        

        # training step
        x, y = batch

        logits = self.forward(x)
        loss = cal_loss(logits, y)

        # logging loss
        self.log("loss", loss, on_epoch=True)

        # tracking accuracy
        preds = torch.max(logits, dim=-1).indices
        labels = y.squeeze()
        self.train_acc(preds, labels)
        self.log("train accuracy", self.train_acc, on_epoch=True, on_step=False)
        
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch 
        logits = self.forward(x)
        loss = cal_loss(logits, y)
        self.log("val_loss", loss)
        
        # accuracy
        preds = torch.max(logits, dim=-1).indices
        labels= y.squeeze()
        self.valid_acc(preds, labels)
        self.log("test_accuracy", self.valid_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.001)
        return opt


    def load_submodules(self, path):
        # loading pretrained submodules
        checkpoint = torch.load(path)
        self.group_devider.load_state_dict(checkpoint['group_devider'])
        self.MAE_encoder.load_state_dict(checkpoint['MAE_encoder'])

        # freeze submodules
        for param in self.MAE_encoder.parameters():
            param.requires_grad = False




if __name__ == "__main__":

    path = '/home/ioannis/Desktop/programming/phd/PCT_Pytorch/data'
    train_loader, valid_loader = get_ModelNet40(path, 'original')

    model = Point_MAE_finetune_pl()
    model.load_submodules("/home/ioannis/Desktop/programming/phd/PointViT/custom_checkpoints/test_ckpt.pt")

    project_name = "FINETUNING POINT_MAE" 
    logger = WandbLogger(project=project_name) 

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=5, logger=logger)
    trainer.fit(model, train_loader, valid_loader)