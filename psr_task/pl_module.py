from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
import scipy
import torch
import torchmetrics

from models import PSRSurfNet


def safe_spearman(gt, pred):
    if np.all(np.isclose(pred, pred[0])):
        return 0
    return scipy.stats.spearmanr(pred, gt).statistic


def rs_metric(reslist, resdict):
    if len(reslist) == 0:
        return 0, 0
    all_lists = np.array(reslist)
    gt, pred = all_lists[:, 0], all_lists[:, 1]
    global_r = safe_spearman(gt, pred)
    local_r = []
    for system, lists in resdict.items():
        lists = np.array(lists)
        gt, pred = lists[:, 0], lists[:, 1]
        local_r.append(safe_spearman(gt, pred))
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

        self.use_graph = hparams.model.use_graph
        self.model = PSRSurfNet(**hparams.model)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def step(self, data):
        name, geom_feats, scores = data.name, data.geom_feats, data.scores
        if name is None:
            return None, None, None, None
        x = (geom_feats, data.graph_feat) if self.use_graph else geom_feats
        output = self(x)
        loss = self.criterion(output, scores)
        return name, loss, output.flatten(), scores

    def training_step(self, batch, batch_idx):
        name, loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))

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

        self.log_dict({"loss/val": loss.item()},
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
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt_params.patience, factor=opt_params.factor, mode='max'),
                     'monitor': "global_r_val",
                     'interval': "epoch",
                     'frequency': 1,
                     "strict": True,
                     'name': "epoch/lr"}
        # return optimizer
        return [optimizer], [scheduler]

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = batch[0]
        batch = batch.to(device)
        return batch
