import logging

import torch
import torch.nn.functional as F
from tllib.alignment.dann import DomainAdversarialLoss, ImageClassifier
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.utils.metric import accuracy
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from medgc_tesis.pipelines.modeling.models.core import DomainAdaptationModel

logger = logging.getLogger(__name__)


class DannModel(DomainAdaptationModel):
    def __init__(self, backbone, bottleneck_dim, lr, momentum, weight_decay, lr_gamma, lr_decay, trade_off):
        super().__init__(
            da_technique="dann",
            model_name=backbone.name,
            classifier=ImageClassifier(
                backbone.model(),
                num_classes=10,
                bottleneck_dim=bottleneck_dim,
                pool_layer=backbone.pool_layer(),
                finetune=True,
            ),
        )
        self.domain_discriminator = DomainDiscriminator(in_feature=self.classifier.features_dim, hidden_size=1024)
        self.domain_adv = DomainAdversarialLoss(self.domain_discriminator)

        self.save_hyperparameters(
            {
                "bottleneck_dim": bottleneck_dim,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "lr_gamma": lr_gamma,
                "lr_decay": lr_decay,
                "trade_off": trade_off,
            }
        )

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_gamma = lr_gamma
        self.lr_decay = lr_decay
        self.trade_off = trade_off

    def configure_optimizers(self):
        optimizer = SGD(
            self.classifier.get_parameters() + self.domain_discriminator.get_parameters(),
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
        y, f = self.classifier(x)
        y_s, _ = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = self.domain_adv(f_s, f_t)
        domain_acc = self.domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * self.trade_off
        cls_acc = accuracy(y_s, labels_s)[0]

        return {
            "loss": loss,
            "cls_acc": cls_acc,
            "domain_acc": domain_acc,
        }

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_cls_acc = torch.stack([x["cls_acc"] for x in outputs]).mean()
        avg_domain_acc = torch.stack([x["domain_acc"] for x in outputs]).mean()

        self.log_dict(
            {
                "train_loss": avg_train_loss,
                "train_cls_acc": avg_cls_acc,
                "train_domain_acc": avg_domain_acc,
            }
        )

    def validation_step(self, batch, batch_idx):
        x_s, labels_s = batch["mnist"]
        x_t, labels_t = batch["tds"]

        x = torch.cat((x_s, x_t), dim=0)
        self.classifier.train()
        y, f = self.classifier(x)
        y_s, _ = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = self.domain_adv(f_s, f_t)
        domain_acc = self.domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * self.trade_off
        cls_acc = accuracy(y_s, labels_s)[0]

        return {
            "loss": loss,
            "cls_acc": cls_acc,
            "domain_acc": domain_acc,
        }

    def validation_epoch_end(self, outputs) -> None:
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        cls_acc = torch.stack([x["cls_acc"] for x in outputs]).mean()
        domain_acc = torch.stack([x["domain_acc"] for x in outputs]).mean()

        self.log_dict(
            {
                "val_loss": loss,
                "val_class_acc": cls_acc,
                "val_domain_acc": domain_acc,
            }
        )
