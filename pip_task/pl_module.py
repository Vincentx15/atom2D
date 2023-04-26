import torch
import pytorch_lightning as pl
import torchmetrics
from models import PIPNet


class PIPModule(pl.LightningModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # self.save_hyperparameters()

        # example
        metric = torchmetrics.Accuracy(task="binary")
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

        self.model = PIPNet()
        self.criterion = torch.nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def step(self, data):
        names_0, _, pos_pairs_cas_arr, neg_pairs_cas_arr, geom_feats_0, geom_feats_1 = data

        all_pairs = torch.cat((pos_pairs_cas_arr, neg_pairs_cas_arr), dim=-3)
        labels = torch.cat((torch.ones(len(pos_pairs_cas_arr)), torch.zeros(len(neg_pairs_cas_arr))))
        output = self(geom_feats_0, geom_feats_1, all_pairs)
        loss = self.criterion(output, labels)
        return loss, output.flatten(), labels.flatten()

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)

        self.log_dict({"loss/train": loss.cpu().detach()},
                      on_step=True, on_epoch=True, prog_bar=True,)

        self.train_accuracy(logits, labels)
        self.log_dict({"acc/train": self.train_accuracy}, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)

        self.log_dict({"loss/val": loss.cpu().detach()},
                      on_step=False, on_epoch=True, prog_bar=True)

        self.val_accuracy(logits, labels)
        self.log_dict({"acc/val": self.val_accuracy}, on_epoch=True)

    def test_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)

        self.log_dict({"loss/test": loss.cpu().detach()},
                      on_step=False, on_epoch=True, prog_bar=True)

        self.test_accuracy(logits, labels)
        self.log_dict({"acc/test": self.test_accuracy}, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        # return [optimizer], [lr_scheduler]
