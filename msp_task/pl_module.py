import torch
import pytorch_lightning as pl
import torchmetrics
from models import MSPSurfNet


class MSPModule(pl.LightningModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # self.save_hyperparameters()

        accuracy = torchmetrics.Accuracy(task="binary")
        self.train_accuracy = accuracy.clone()
        self.val_accuracy = accuracy.clone()
        self.test_accuracy = accuracy.clone()

        auroc = torchmetrics.AUROC(task="binary")
        self.train_auroc = auroc.clone()
        self.val_auroc = auroc.clone()
        self.test_auroc = auroc.clone()

        self.model = MSPSurfNet()
        self.criterion = torch.nn.BCELoss()

    def forward(self, x):
        return self.model(*x)

    def step(self, data):
        name, geom_feats, coords, label = data[0]
        if name is None:
            return None, None, None

        label = torch.tensor([int(label)]).float().to(self.device)
        geom_feats = [[y.to(self.device) for y in x] for x in geom_feats]
        coords = [x.to(self.device) for x in coords]
        output = self((geom_feats, coords))
        loss = self.criterion(output, label)
        return loss, output.flatten(), label

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/train": loss.cpu().detach()},
                      on_step=True, on_epoch=True, prog_bar=True, batch_size=len(logits))

        self.train_accuracy(logits, labels)
        self.train_auroc(logits, labels)
        self.log_dict({"acc/train": self.train_accuracy, "auroc/train": self.train_auroc}, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/val": loss.cpu().detach()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

        self.val_accuracy(logits, labels)
        self.val_auroc(logits, labels)
        self.log_dict({"acc/val": self.val_accuracy, "auroc/val": self.val_auroc}, on_epoch=True)

    def test_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/test": loss.cpu().detach()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

        self.test_accuracy(logits, labels)
        self.test_auroc(logits, labels)
        self.log_dict({"acc/test": self.test_accuracy, "auroc/test": self.test_auroc}, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        # return [optimizer], [lr_scheduler]
