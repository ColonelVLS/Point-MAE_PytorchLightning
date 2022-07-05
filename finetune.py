from model import Point_MAE_pl
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy
from dl_lib.datasets.ready_datasets import get_ModelNet40

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

class Point_MAE_finetune_pl(Point_MAE_pl):
    def __init__(self):
        super().__init__()

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):        

        # training step
        x, y = batch

        logits = self.net(x.permute(0,2,1))
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
        logits = self.net(x.permute(0, 2, 1))
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


if __name__=="__main__":

    path = '/home/ioannis/Desktop/programming/phd/PCT_Pytorch/data'
    train_loader, valid_loader = get_ModelNet40(path, 'original')

    checkpoint_path = "/home/ioannis/Desktop/programming/phd/PointViT/Point-MAE/1lkzfn0m/checkpoints/epoch=73-step=97014.ckpt"
    model = Point_MAE_finetune_pl.load_from_checkpoint(checkpoint_path)

    trainer = pl.Trainer()
    
    project_name = "FINETUNING POINT_MAE" 
    logger = WandbLogger(project=project_name) 

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=300, logger=logger)
    trainer.fit(model, train_loader, valid_loader)