import torch
import pytorch_lightning as pl
import torch.nn as nn
# import torchmetrics

from sklearn.metrics import roc_auc_score, accuracy_score
from psr_task.models import PSRSurfNet


def compute_accuracy(predictions, labels):
    # Convert predictions to binary labels (0 or 1)
    predicted_labels = torch.argmax(predictions, dim=1)
    predicted_labels = predicted_labels.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    accuracy = accuracy_score(y_true=labels, y_pred=predicted_labels)
    return accuracy


def compute_auroc(predictions, labels):
    labels = labels.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    try:
        # No need to softmax here, since the only issue is to sort for auroc
        predictions /= predictions.sum(axis=1)[:,None]
        auroc = roc_auc_score(y_true=labels, y_score=predictions, multi_class='ovr') # TODO : debug
        # TODO : this is not trivial in the multiclass case https://github.com/scikit-learn/scikit-learn/issues/24636
        # probably, we need to one hot encode the labels.
        return auroc
    except ValueError as e:
        print("Auroc computation failed, ", e)
        return 0.5


class HoloProtPLModule(pl.LightningModule):

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.use_graph = hparams.model.use_graph or hparams.model.use_graph_only
        self.use_graph_only = hparams.model.use_graph_only
        self.model = PSRSurfNet(**hparams.model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        if not hasattr(batch, "graph"):
            return None, None, None
        labels = batch.y.flatten()
        output = self(batch)
        loss = self.criterion(output, labels)
        if torch.isnan(loss).any():
            return None, None, None
        return loss, output, labels.flatten()

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))

        acc = compute_accuracy(logits, labels)
        # auroc = compute_auroc(logits, labels)

        self.log_dict({"acc/train": acc}, on_epoch=True, batch_size=len(logits))

        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)
        if loss is None or logits.isnan().any() or labels.isnan().any():
            print("validation step skipped!")
            self.log("acc_cal", 0., prog_bar=True, on_step=False, on_epoch=True, logger=False, batch_size=1)
            return None

        self.log_dict({"loss/val": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

        acc = compute_accuracy(logits, labels)
        # auroc = compute_auroc(logits, labels)

        self.log_dict({"acc/val": acc}, on_epoch=True)
        self.log("acc_val", acc, prog_bar=True, on_step=False, on_epoch=True, logger=False)

    def test_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)
        if loss is None or logits.isnan().any() or labels.isnan().any():
            self.log("acc/test", 0.5, on_epoch=True)
            return None

        self.log_dict({"loss/test": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

        acc = compute_accuracy(logits, labels)
        # auroc = compute_auroc(logits, labels)

        self.log_dict({"acc/test": acc}, on_epoch=True)

    def configure_optimizers(self):
        opt_params = self.hparams.hparams.optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=opt_params.lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt_params.patience,
                                                                             factor=opt_params.factor, mode='max'),
                     'monitor': "acc_val",
                     'interval': "epoch",
                     'frequency': 1,
                     "strict": True,
                     'name': "epoch/lr"}
        return [optimizer], [scheduler]

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = batch.to(device)
        return batch
