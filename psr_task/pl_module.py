import torch
import pytorch_lightning as pl
import torchmetrics
from models import PSRSurfNet


class PSRModule(pl.LightningModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # self.save_hyperparameters()

        mean = torchmetrics.MeanMetric()
        self.train_accuracy = mean.clone()
        self.val_accuracy = mean.clone()
        self.test_accuracy = mean.clone()

        self.model = PSRSurfNet()
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def step(self, data):
        name, geom_feats, scores = data[0]
        if name is None:
            return None, None, None

        output = self(geom_feats)
        loss = self.criterion(output, scores)
        return loss, output.flatten(), scores

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/train": loss.cpu().detach()},
                      on_step=True, on_epoch=True, prog_bar=True, batch_size=len(logits))

        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/val": loss.cpu().detach()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

    def test_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/test": loss.cpu().detach()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        # return [optimizer], [lr_scheduler]

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if batch[0][0] is None:
            return batch

        name, geom_feats, scores = batch[0]
        geom_feats = [x.to(self.device) for x in geom_feats]
        scores = scores.to(self.device)
        batch = [(name, geom_feats, scores)]
        return batch
