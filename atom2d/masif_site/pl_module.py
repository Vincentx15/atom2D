import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from masif_site.models import MasifSiteNet
from pip_task.pl_module import compute_auroc, compute_accuracy


def masif_site_loss(preds, labels):
    # Inspired from dmasif
    pos_preds = preds[labels == 1]
    pos_labels = torch.ones_like(pos_preds)
    neg_preds = [preds][labels == 0]
    neg_labels = torch.zeros_like(pos_preds)
    n_points_sample = min(len(pos_labels), len(neg_labels))
    pos_indices = torch.randperm(len(pos_labels))[:n_points_sample]
    neg_indices = torch.randperm(len(neg_labels))[:n_points_sample]
    pos_preds = pos_preds[pos_indices]
    pos_labels = pos_labels[pos_indices]
    neg_preds = neg_preds[neg_indices]
    neg_labels = neg_labels[neg_indices]
    preds_concat = torch.cat([pos_preds, neg_preds])
    labels_concat = torch.cat([pos_labels, neg_labels])
    loss = F.binary_cross_entropy_with_logits(preds_concat, labels_concat)
    return loss, preds_concat, labels_concat


class MasifSiteModule(pl.LightningModule):

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = MasifSiteNet(**hparams.model)

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        if (not hasattr(batch, "surface") and
                not hasattr(batch,"graph")):  # if no surface and no graph, then the full batch was filtered out
            return None, None, None
        labels = batch.labels.flatten()
        output = self(batch).flatten()
        loss, preds_concat, labels_concat = masif_site_loss(output, labels)
        return loss, preds_concat, labels_concat

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))
        acc = compute_accuracy(logits, labels)
        auroc = compute_auroc(logits, labels)
        self.log_dict({"acc/train": acc, "auroc/train": auroc}, on_epoch=True, batch_size=len(logits))
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)
        if loss is None or logits.isnan().any() or labels.isnan().any():
            print("validation step skipped!")
            self.log("auroc_val", 0.5, prog_bar=True, on_step=False, on_epoch=True, logger=False)
            return None
        self.log_dict({"loss/val": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        acc = compute_accuracy(logits, labels)
        auroc = compute_auroc(logits, labels)
        self.log_dict({"acc/val": acc, "auroc/val": auroc}, on_epoch=True)
        self.log("auroc_val", auroc, prog_bar=True, on_step=False, on_epoch=True, logger=False)

    def test_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)
        if loss is None or logits.isnan().any() or labels.isnan().any():
            self.log("acc/test", 0.5, on_epoch=True)
            return None
        self.log_dict({"loss/test": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        acc = compute_accuracy(logits, labels)
        auroc = compute_auroc(logits, labels)
        self.log_dict({"acc/test": acc, "auroc/test": auroc}, on_epoch=True)

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
        batch = batch.to(device)
        return batch