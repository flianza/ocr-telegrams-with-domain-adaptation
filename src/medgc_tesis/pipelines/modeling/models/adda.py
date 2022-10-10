import copy
import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tllib.alignment.adda import ImageClassifier
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.modules.grl import WarmStartGradientReverseLayer
from tllib.translation.cyclegan.util import set_requires_grad
from tllib.utils.metric import accuracy
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from medgc_tesis.pipelines.modeling.models.core import DomainAdaptationModel
from medgc_tesis.pipelines.modeling.models.data import DomainAdaptationDataModule

logger = logging.getLogger(__name__)


class ImageClassifierAddaModel(pl.LightningModule):
    def __init__(self, backbone, bottleneck_dim, lr, momentum, weight_decay, lr_gamma, lr_decay) -> None:
        super().__init__()
        self.classifier = ImageClassifier(
            backbone.model(),
            num_classes=10,
            bottleneck_dim=bottleneck_dim,
            pool_layer=backbone.pool_layer(),
            finetune=True,
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

        y_s, f_s = self.classifier(x_s)

        loss = F.cross_entropy(y_s, labels_s)
        cls_acc = accuracy(y_s, labels_s)[0]

        return {
            "loss": loss,
            "cls_acc": cls_acc,
        }

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        cls_acc = torch.stack([x["cls_acc"] for x in outputs]).mean()

        self.log_dict(
            {
                "train_loss": train_loss,
                "train_cls_acc": cls_acc,
            }
        )

    def validation_step(self, batch, batch_idx):
        x_s, labels_s = batch["mnist"]

        y_s = self.classifier(x_s)

        loss = F.cross_entropy(y_s, labels_s)
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
                "val_cls_acc": cls_acc,
            }
        )


class AddaModel(DomainAdaptationModel):
    def __init__(
        self,
        backbone,
        bottleneck_dim,
        lr,
        pretrain_lr,
        pretrain_epochs,
        momentum,
        weight_decay,
        lr_gamma,
        lr_decay,
    ):
        super().__init__(da_technique="adda", model_name=backbone.name, classifier=None)
        self.save_hyperparameters(
            {
                "bottleneck_dim": bottleneck_dim,
                "lr": lr,
                "pretrain_lr": pretrain_lr,
                "pretrain_epochs": pretrain_epochs,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "lr_gamma": lr_gamma,
                "lr_decay": lr_decay,
            }
        )

        self.backbone = backbone
        self.bottleneck_dim = bottleneck_dim
        self.lr = lr
        self.pretrain_lr = pretrain_lr
        self.pretrain_epochs = pretrain_epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_gamma = lr_gamma
        self.lr_decay = lr_decay

        self._pretrain_model()

        self.domain_discriminator = DomainDiscriminator(in_feature=self.classifier.features_dim, hidden_size=1024)

        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=2.0, max_iters=1000, auto_step=True)
        self.domain_adv = DomainAdversarialLoss(self.domain_discriminator, grl=self.grl)

    def _pretrain_model(self):
        dm = DomainAdaptationDataModule(transform=self.backbone.data_transform())

        model = ImageClassifierAddaModel(
            self.backbone,
            self.bottleneck_dim,
            self.pretrain_lr,
            self.momentum,
            self.weight_decay,
            self.lr_gamma,
            self.lr_decay,
        )

        trainer = Trainer(
            accelerator="gpu",
            gpus=0,
            max_epochs=self.pretrain_epochs,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)],
            deterministic=True,
            logger=None,
            enable_checkpointing=False,
        )

        trainer.fit(model, datamodule=dm)

        self.source_classifier = copy.deepcopy(model.classifier)
        self.classifier = copy.deepcopy(model.classifier)

    def configure_optimizers(self):
        optimizer = SGD(
            self.classifier.get_parameters(optimize_head=False) + self.domain_discriminator.get_parameters(),
            self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )
        lr_scheduler = LambdaLR(optimizer, lambda x: self.lr * (1.0 + self.lr_gamma * float(x)) ** (-self.lr_decay))
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_fit_start(self) -> None:
        # freeze source classifier
        set_requires_grad(self.source_classifier, False)
        self.source_classifier.freeze_bn()

    def training_step(self, batch, batch_idx):
        x_s, _ = batch["mnist"]
        x_t, _ = batch["tds"]

        _, f_s = self.source_classifier(x_s)
        _, f_t = self.classifier(x_t)

        loss = self.domain_adv(f_s, f_t)
        domain_acc = self.domain_adv.domain_discriminator_accuracy

        return {
            "loss": loss,
            "domain_acc": domain_acc,
        }

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        domain_acc = torch.stack([x["domain_acc"] for x in outputs]).mean()

        self.log_dict(
            {
                "train_loss": loss,
                "train_domain_acc": domain_acc,
            }
        )

    def on_validation_start(self) -> None:
        self.source_classifier.train()
        self.classifier.train()

        # freeze source classifier
        set_requires_grad(self.source_classifier, False)
        self.source_classifier.freeze_bn()

    def validation_step(self, batch, batch_idx):
        x_s, _ = batch["mnist"]
        x_t, _ = batch["tds"]

        _, f_s = self.source_classifier(x_s)
        _, f_t = self.classifier(x_t)

        loss = self.domain_adv(f_s, f_t)
        domain_acc = self.domain_adv.domain_discriminator_accuracy

        return {
            "loss": loss,
            "domain_acc": domain_acc,
        }

    def validation_epoch_end(self, outputs) -> None:
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        domain_acc = torch.stack([x["domain_acc"] for x in outputs]).mean()

        self.log_dict(
            {
                "val_loss": loss,
                "val_domain_acc": domain_acc,
            }
        )
