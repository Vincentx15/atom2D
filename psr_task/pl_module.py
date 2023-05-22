from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
import scipy
import torch
import torchmetrics

from models import PSRSurfNet


def rs_metric(reslist, resdict):
    all_lists = np.array(reslist)
    global_r = scipy.stats.spearmanr(all_lists, axis=0).statistic
    local_r = []
    for system, lists in resdict.items():
        lists = np.array(lists)
        r = scipy.stats.spearmanr(lists, axis=0).statistic
        local_r.append(r)
    local_r = float(np.mean(local_r))
    return global_r, local_r


class PSRModule(pl.LightningModule):

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

        mean = torchmetrics.MeanMetric()
        self.train_accuracy = mean.clone()
        self.val_accuracy = mean.clone()
        self.test_accuracy = mean.clone()

        self.val_reslist = list()
        self.val_resdict = defaultdict(list)

        self.test_reslist = list()
        self.test_resdict = defaultdict(list)

        self.model = PSRSurfNet(**hparams.model)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def step(self, data):
        name, geom_feats, scores = data[0]
        if name is None:
            return None, None, None, None

        output = self(geom_feats)
        loss = self.criterion(output, scores)
        return name, loss, output.flatten(), scores

    def training_step(self, batch, batch_idx):
        name, loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/train": loss.cpu().detach()},
                      on_step=True, on_epoch=True, prog_bar=True, batch_size=len(logits))

        return loss

    def validation_step(self, batch, batch_idx: int):
        name, loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        scores = logits.item()
        output = labels.item()
        reslist = [output, scores]
        self.val_reslist.append(reslist)
        self.val_resdict[name[:4]].append(reslist)

        self.log_dict({"loss/val": loss.cpu().detach()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

    def test_step(self, batch, batch_idx: int):
        name, loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        scores = logits.item()
        output = labels.item()
        reslist = [output, scores]
        self.test_reslist.append(reslist)
        self.test_resdict[name[:4]].append(reslist)

        self.log_dict({"loss/test": loss.cpu().detach()},
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
