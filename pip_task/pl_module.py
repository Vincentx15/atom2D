import torch
import pytorch_lightning as pl
# import torchmetrics
from sklearn.metrics import roc_auc_score
from models import PIPNet


def compute_accuracy(predictions, labels):
    # Convert predictions to binary labels (0 or 1)
    predicted_labels = torch.round(predictions)
    # Compare predicted labels with ground truth labels
    correct_count = (predicted_labels == labels).sum().item()
    total_count = labels.size(0)
    # Compute accuracy
    accuracy = correct_count / total_count
    return accuracy


def compute_auroc(predictions, labels):
    labels = labels.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    try:
        auroc = roc_auc_score(y_true=labels, y_score=predictions)
        return auroc
    except ValueError as e:
        print("Auroc computation failed, ", e)
        return 0.5


class PIPModule(pl.LightningModule):

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

        # accuracy = torchmetrics.Accuracy(task="binary")
        # self.train_accuracy = accuracy.clone()
        # self.val_accuracy = accuracy.clone()
        # self.test_accuracy = accuracy.clone()

        # auroc = torchmetrics.AUROC(task="binary")
        # self.train_auroc = auroc.clone()
        # self.val_auroc = auroc.clone()
        # self.test_auroc = auroc.clone()

        self.use_graph = hparams.model.use_graph or hparams.model.use_graph_only
        self.model = PIPNet(**hparams.model)
        # self.criterion = torch.nn.BCELoss()
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hparams.model.pos_weight]))

    def forward(self, x):
        return self.model(*x)

    def step(self, data):
        names_0, pos_pairs_cas_arr, neg_pairs_cas_arr = (data.name1, data.pos_stack, data.neg_stack) if "name1" in data else (None, None, None)
        if names_0 is None or pos_pairs_cas_arr.numel() == 0 or neg_pairs_cas_arr.numel() == 0:
            return None, None, None
        geom_feats_1, geom_feats_2 = data.geom_feats_1, data.geom_feats_2
        x_1, x_2 = ((geom_feats_1, data.graph_1), (geom_feats_2, data.graph_2)) if self.use_graph else (geom_feats_1, geom_feats_2)
        all_pairs = torch.cat((pos_pairs_cas_arr, neg_pairs_cas_arr), dim=-3)
        labels = torch.cat((torch.ones(len(pos_pairs_cas_arr)), torch.zeros(len(neg_pairs_cas_arr)))).to(self.device)

        output = self((x_1, x_2, all_pairs))
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

        acc = compute_accuracy(logits, labels)
        auroc = compute_auroc(logits, labels)

        # self.train_accuracy(logits, labels)
        # self.train_auroc(logits, labels)
        # self.log_dict({"acc/train": self.train_accuracy, "auroc/train": self.train_auroc}, on_epoch=True)
        self.log_dict({"acc/train": acc, "auroc/train": auroc}, on_epoch=True, batch_size=len(logits))

        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/val": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

        acc = compute_accuracy(logits, labels)
        auroc = compute_auroc(logits, labels)

        # self.val_accuracy(logits, labels)
        # self.val_auroc(logits, labels)
        # self.log_dict({"acc/val": self.val_accuracy, "auroc/val": self.val_auroc}, on_epoch=True)
        # self.log("auroc_val", self.val_auroc, prog_bar=True, on_step=False, on_epoch=True, logger=False)
        self.log_dict({"acc/val": acc, "auroc/val": auroc}, on_epoch=True)
        self.log("auroc_val", auroc, prog_bar=True, on_step=False, on_epoch=True, logger=False)

    def test_step(self, batch, batch_idx: int):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/test": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

        acc = compute_accuracy(logits, labels)
        auroc = compute_auroc(logits, labels)

        # self.test_accuracy(logits, labels)
        # self.test_auroc(logits, labels)
        # self.log_dict({"acc/test": self.test_accuracy, "auroc/test": self.test_auroc}, on_epoch=True)
        self.log_dict({"acc/test": acc, "auroc/test": auroc}, on_epoch=True)

    def configure_optimizers(self):
        opt_params = self.hparams.hparams.optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=opt_params.lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt_params.patience, factor=opt_params.factor, mode='max'),
                     'monitor': "auroc_val",
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
