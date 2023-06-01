import torch
import pytorch_lightning as pl
import torchmetrics
from models import PIPNet


class PIPModule(pl.LightningModule):

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

        self.model = PIPNet(**hparams.model)
        self.criterion = torch.nn.BCELoss()

    def forward(self, x):
        return self.model(*x)

    def step(self, data):
        names_0, _, pos_pairs_cas_arr, neg_pairs_cas_arr, geom_feats_0, geom_feats_1 = data[0]
        if names_0 is None or pos_pairs_cas_arr.numel() == 0 or neg_pairs_cas_arr.numel() == 0:
            return None, None, None

        all_pairs = torch.cat((pos_pairs_cas_arr, neg_pairs_cas_arr), dim=-3)
        labels = torch.cat((torch.ones(len(pos_pairs_cas_arr)), torch.zeros(len(neg_pairs_cas_arr)))).to(self.device)
        output = self((geom_feats_0, geom_feats_1, all_pairs))
        loss = self.criterion(output, labels)
        return loss, output.flatten(), labels.flatten()

    def training_step(self, batch, batch_idx):
        # t0 = time.perf_counter()
        loss, logits, labels = self.step(batch)
        # torch.cuda.synchronize()
        # print(f"Time for one step  : {time.perf_counter()-t0}")
        if loss is None:
            return None

        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))

        self.train_accuracy(logits, labels)
        self.train_auroc(logits, labels)
        self.log_dict({"acc/train": self.train_accuracy, "auroc/train": self.train_auroc}, on_epoch=True)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.hparams.optimizer.lr)
        return optimizer
        # return [optimizer], [lr_scheduler]
