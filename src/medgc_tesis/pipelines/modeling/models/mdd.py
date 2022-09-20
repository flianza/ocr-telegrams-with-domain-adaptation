import logging

import torch
import torch.nn.functional as F
from tllib.alignment.mdd import (
    ClassificationMarginDisparityDiscrepancy as MarginDisparityDiscrepancy,
)
from tllib.alignment.mdd import ImageClassifier
from tllib.utils.metric import accuracy
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from medgc_tesis.pipelines.modeling.models.core import DomainAdaptationModel

logger = logging.getLogger(__name__)


class MddModel(DomainAdaptationModel):
    def __init__(self, backbone, bottleneck_dim, lr, momentum, weight_decay, lr_gamma, lr_decay, margin, trade_off):
        super().__init__(
            name="mdd",
            classifier=ImageClassifier(
                backbone.model(),
                num_classes=10,
                bottleneck_dim=bottleneck_dim,
                pool_layer=backbone.pool_layer(),
                finetune=True,
            ),
        )
        self.mdd = MarginDisparityDiscrepancy(margin)

        self.save_hyperparameters(
            {
                "bottleneck_dim": bottleneck_dim,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "lr_gamma": lr_gamma,
                "lr_decay": lr_decay,
                "margin": margin,
                "trade_off": trade_off,
            }
        )

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_gamma = lr_gamma
        self.lr_decay = lr_decay
        self.margin = margin
        self.trade_off = trade_off

    def configure_optimizers(self):
        optimizer = SGD(
            self.classifier.get_parameters(),
            self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )
        lr_scheduler = LambdaLR(
            optimizer,
            lambda x: self.lr * (1.0 + self.lr_gamma * float(x)) ** (-self.lr_decay),
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        x_s, labels_s = batch["mnist"]
        x_t, labels_t = batch["tds"]

        x = torch.cat((x_s, x_t), dim=0)
        outputs, outputs_adv = self.classifier(x)
        y_s, y_t = outputs.chunk(2, dim=0)
        y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = -self.mdd(y_s, y_s_adv, y_t, y_t_adv)
        loss = cls_loss + transfer_loss * self.trade_off
        self.classifier.step()
        cls_acc = accuracy(y_s, labels_s)[0]

        return {
            "loss": loss,
            "cls_acc": cls_acc,
        }

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_cls_acc = torch.stack([x["cls_acc"] for x in outputs]).mean()

        self.log_dict(
            {
                "train_loss": avg_train_loss,
                "train_cls_acc": avg_cls_acc,
            }
        )

    def validation_step(self, batch, batch_idx):
        x_s, labels_s = batch["mnist"]
        x_t, labels_t = batch["tds"]

        x = torch.cat((x_s, x_t), dim=0)
        self.classifier.train()
        outputs, outputs_adv = self.classifier(x)
        y_s, y_t = outputs.chunk(2, dim=0)
        y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = -self.mdd(y_s, y_s_adv, y_t, y_t_adv)
        loss = cls_loss + transfer_loss * self.trade_off
        cls_acc = accuracy(y_s, labels_s)[0]

        return {
            "loss": loss,
            "cls_acc": cls_acc,
        }

    def validation_epoch_end(self, outputs) -> None:
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        cls_acc = torch.stack([x["cls_acc"] for x in outputs]).mean()

        self.log_dict(
            {
                "val_loss": loss,
                "val_class_acc": cls_acc,
            }
        )
