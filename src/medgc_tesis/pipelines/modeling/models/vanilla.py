import logging

import torch
import torch.nn.functional as F
from tllib.modules.classifier import Classifier
from tllib.utils.metric import accuracy
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from medgc_tesis.pipelines.modeling.models.core import DomainAdaptationModel

logger = logging.getLogger(__name__)


class SourceOnlyModel(DomainAdaptationModel):
    def __init__(self, backbone, lr, momentum, weight_decay, lr_gamma, lr_decay):
        super().__init__(
            name="source_only",
            classifier=Classifier(
                backbone.model(),
                num_classes=10,
                pool_layer=backbone.pool_layer(),
                finetune=True,
            ),
        )

        self.save_hyperparameters(
            {
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "lr_gamma": lr_gamma,
                "lr_decay": lr_decay,
            }
        )

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_gamma = lr_gamma
        self.lr_decay = lr_decay

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
        x_t, _ = batch["tds"]

        y_s, f_s = self.classifier(x_s)

        loss = F.cross_entropy(y_s, labels_s)
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

        y_s = self.classifier(x_s)

        loss = F.cross_entropy(y_s, labels_s)
        cls_acc = accuracy(y_s, labels_s)[0]

        return {
            "val_loss_step": loss,
            "val_class_acc_step": cls_acc,
        }

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["val_loss_step"] for x in outputs]).mean()
        avg_class_acc = torch.stack([x["val_class_acc_step"] for x in outputs]).mean()

        self.log_dict(
            {
                "val_loss": avg_loss,
                "val_class_acc": avg_class_acc,
            }
        )


class TargetOnlyModel(DomainAdaptationModel):
    def __init__(self, backbone, lr, momentum, weight_decay, lr_gamma, lr_decay):
        super().__init__(
            name="target_only",
            classifier=Classifier(
                backbone.model(),
                num_classes=10,
                pool_layer=backbone.pool_layer(),
                finetune=True,
            ),
        )

        self.save_hyperparameters(
            {
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "lr_gamma": lr_gamma,
                "lr_decay": lr_decay,
            }
        )

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_gamma = lr_gamma
        self.lr_decay = lr_decay

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

        y_t, f_t = self.classifier(x_t)

        loss = F.cross_entropy(y_t, labels_t)
        cls_acc = accuracy(y_t, labels_t)[0]

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

        y_t = self.classifier(x_t)

        loss = F.cross_entropy(y_t, labels_t)
        cls_acc = accuracy(y_t, labels_t)[0]

        return {
            "val_loss_step": loss,
            "val_class_acc_step": cls_acc,
        }

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["val_loss_step"] for x in outputs]).mean()
        avg_class_acc = torch.stack([x["val_class_acc_step"] for x in outputs]).mean()

        self.log_dict(
            {
                "val_loss": avg_loss,
                "val_class_acc": avg_class_acc,
            }
        )
