import torch
import pytorch_lightning as pl
import torchmetrics
from models import MSPSurfNet


class MSPModule(pl.LightningModule):

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

        accuracy = torchmetrics.Accuracy(task="binary")
        self.train_accuracy = accuracy.clone()
        self.val_accuracy = accuracy.clone()
        self.test_accuracy = accuracy.clone()

        auroc = torchmetrics.AUROC(task="binary")
        self.train_auroc = auroc.clone()
        self.val_auroc = auroc.clone()
        self.test_auroc = auroc.clone()

        self.use_graph = hparams.model.use_graph or hparams.model.use_graph_only
        self.use_graph_only = hparams.model.use_graph_only
        self.model = MSPSurfNet(**hparams.model)
        # self.criterion = torch.nn.BCELoss()
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hparams.model.pos_weight]))

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        filtered_batch = [data for data in batch if "names" in data]
        if len(filtered_batch) == 0:
            return None, None, None
        label = torch.cat([data.label for data in filtered_batch])
        output = self(filtered_batch)
        loss = self.criterion(output, label)
        return loss, output.flatten(), label

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))

        self.train_accuracy(logits, labels)
        self.train_auroc(logits, labels)
        self.log_dict({"acc/train": self.train_accuracy, "auroc/train": self.train_auroc}, on_epoch=True,
                      batch_size=len(logits))

        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/val": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

        self.val_accuracy(logits, labels)
        self.val_auroc(logits, labels)
        self.log_dict({"acc/val": self.val_accuracy, "auroc/val": self.val_auroc}, on_epoch=True)
        self.log("auroc_val", self.val_auroc, prog_bar=True, on_step=False, on_epoch=True, logger=False)

    def test_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/test": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

        self.test_accuracy(logits, labels)
        self.test_auroc(logits, labels)
        self.log_dict({"acc/test": self.test_accuracy, "auroc/test": self.test_auroc}, on_epoch=True)

    def configure_optimizers(self):
        opt_params = self.hparams.hparams.optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=opt_params.lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt_params.patience,
                                                                             factor=opt_params.factor, mode='max'),
                     'monitor': "auroc_val",
                     'interval': "epoch",
                     'frequency': 1,
                     "strict": True,
                     'name': "epoch/lr"}
        # return optimizer
        return [optimizer], [scheduler]

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = [data.to(device) for data in batch]
        return batch
