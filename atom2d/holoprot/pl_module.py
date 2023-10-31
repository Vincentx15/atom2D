from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
import scipy
import torch
import torchmetrics

from psr_task.models import PSRSurfNet

class HoloProtPLModule(pl.LightningModule):

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

        mean = torchmetrics.MeanMetric()
        self.train_accuracy = mean.clone()
        self.val_accuracy = mean.clone()
        self.test_accuracy = mean.clone()
        self.model = PSRSurfNet(**hparams.model)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        if not hasattr(batch, "surface") and not hasattr(batch, "graph"):  # if no surface and no graph, then the full batch was filtered out
            return None, None, None, None

        names = batch.name
        scores = batch.scores.flatten()
        output = self(batch).flatten()
        loss = self.criterion(output, scores)
        return names, loss, output.flatten(), scores

    def training_step(self, batch, batch_idx):
        names, loss, logits, labels = self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))
        return loss

    def validation_step(self, batch, batch_idx: int):
        names, loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        scores = logits.detach().cpu().numpy()
        outputs = labels.detach().cpu().numpy()
        for name, output, score in zip(names, outputs, scores):
            reslist = [output, score]
            self.val_reslist.append(reslist)
            self.val_resdict[name[:4]].append(reslist)

        self.log_dict({"loss/val": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

    def test_step(self, batch, batch_idx: int):
        names, loss, logits, labels = self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/test": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

    def on_validation_epoch_end(self):
        # Do the computation over the epoch results and reset the epoch variables
        global_r, local_r = rs_metric(self.val_reslist, self.val_resdict)
        self.val_reslist = list()
        self.val_resdict = defaultdict(list)

        print(f" Global R validation: {global_r}")
        print(f" Local R validation : {local_r}")
        self.log_dict({"global_r/val": global_r})
        self.log_dict({"local_r/val": local_r})
        self.log("global_r_val", global_r, prog_bar=True, on_step=False, on_epoch=True, logger=False)

    def on_test_epoch_end(self):
        # Do the computation over the epoch results and reset the epoch variables
        global_r, local_r = rs_metric(self.test_reslist, self.test_resdict)
        self.test_reslist = list()
        self.test_resdict = defaultdict(list)

        print(f" Global R test: {global_r}")
        print(f" Local R test : {local_r}")
        self.log_dict({"global_r/test": global_r})
        self.log_dict({"local_r/test": local_r})

    def configure_optimizers(self):
        opt_params = self.hparams.hparams.optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=opt_params.lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt_params.patience,
                                                                             factor=opt_params.factor, mode='max'),
                     'monitor': "global_r_val",
                     'interval': "epoch",
                     'frequency': 1,
                     "strict": True,
                     'name': "epoch/lr"}
        # return optimizer
        return [optimizer], [scheduler]

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = batch.to(device)
        return batch
