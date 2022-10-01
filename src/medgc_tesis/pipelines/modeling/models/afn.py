import logging

import torch
import torch.nn.functional as F
from tllib.modules.entropy import entropy
from tllib.normalization.afn import AdaptiveFeatureNorm, ImageClassifier
from tllib.utils.metric import accuracy
from torch.optim import SGD

from medgc_tesis.pipelines.modeling.models.core import DomainAdaptationModel

logger = logging.getLogger(__name__)


class AfnModel(DomainAdaptationModel):
    def __init__(
        self,
        backbone,
        bottleneck_dim,
        num_blocks,
        dropout_p,
        delta,
        lr,
        weight_decay,
        trade_off_norm,
        trade_off_entropy=None,
    ):
        super().__init__(
            da_technique="afn",
            model_name=backbone.name,
            classifier=ImageClassifier(
                backbone.model(),
                num_classes=10,
                bottleneck_dim=bottleneck_dim,
                num_blocks=num_blocks,
                dropout_p=dropout_p,
                pool_layer=backbone.pool_layer(),
                finetune=True,
            ),
        )
        self.adaptive_feature_norm = AdaptiveFeatureNorm(delta)

        self.save_hyperparameters(
            {
                "bottleneck_dim": bottleneck_dim,
                "num_blocks": num_blocks,
                "dropout_p": dropout_p,
                "delta": delta,
                "lr": lr,
                "weight_decay": weight_decay,
                "trade_off_norm": trade_off_norm,
                "trade_off_entropy": trade_off_entropy,
            }
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.trade_off_norm = trade_off_norm
        self.trade_off_entropy = trade_off_entropy

    def configure_optimizers(self):
        optimizer = SGD(
            self.classifier.get_parameters(),
            self.lr,
            weight_decay=self.weight_decay,
        )

        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        x_s, labels_s = batch["mnist"]
        x_t, labels_t = batch["tds"]

        # compute output
        y_s, f_s = self.classifier(x_s)
        y_t, f_t = self.classifier(x_t)

        # classification loss
        cls_loss = F.cross_entropy(y_s, labels_s)
        norm_loss = self.adaptive_feature_norm(f_s) + self.adaptive_feature_norm(f_t)
        loss = cls_loss + norm_loss * self.trade_off_norm

        # using entropy minimization
        if self.trade_off_entropy:
            y_t = F.softmax(y_t, dim=1)
            entropy_loss = entropy(y_t, reduction="mean")
            loss += entropy_loss * self.trade_off_entropy

        cls_acc = accuracy(y_s, labels_s)[0]
        src_feature_norm = f_s.norm(p=2, dim=1).mean()
        tgt_feature_norm = f_t.norm(p=2, dim=1).mean()

        return {
            "loss": loss,
            "cls_acc": cls_acc,
            "src_feature_norm": src_feature_norm,
            "tgt_feature_norm": tgt_feature_norm,
        }

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        cls_acc = torch.stack([x["cls_acc"] for x in outputs]).mean()
        src_feature_norm = torch.stack([x["src_feature_norm"] for x in outputs]).mean()
        tgt_feature_norm = torch.stack([x["tgt_feature_norm"] for x in outputs]).mean()

        self.log_dict(
            {
                "train_loss": train_loss,
                "train_cls_acc": cls_acc,
                "train_src_feature_norm": src_feature_norm,
                "train_tgt_feature_norm": tgt_feature_norm,
            }
        )

    def validation_step(self, batch, batch_idx):
        x_s, labels_s = batch["mnist"]
        x_t, labels_t = batch["tds"]

        # compute output
        self.classifier.train()
        y_s, f_s = self.classifier(x_s)
        y_t, f_t = self.classifier(x_t)

        # classification loss
        cls_loss = F.cross_entropy(y_s, labels_s)
        norm_loss = self.adaptive_feature_norm(f_s) + self.adaptive_feature_norm(f_t)
        loss = cls_loss + norm_loss * self.trade_off_norm

        # using entropy minimization
        if self.trade_off_entropy:
            y_t = F.softmax(y_t, dim=1)
            entropy_loss = entropy(y_t, reduction="mean")
            loss += entropy_loss * self.trade_off_entropy

        cls_acc = accuracy(y_s, labels_s)[0]
        src_feature_norm = f_s.norm(p=2, dim=1).mean()
        tgt_feature_norm = f_t.norm(p=2, dim=1).mean()

        return {
            "loss": loss,
            "cls_acc": cls_acc,
            "src_feature_norm": src_feature_norm,
            "tgt_feature_norm": tgt_feature_norm,
        }

    def validation_epoch_end(self, outputs) -> None:
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        cls_acc = torch.stack([x["cls_acc"] for x in outputs]).mean()
        src_feature_norm = torch.stack([x["src_feature_norm"] for x in outputs]).mean()
        tgt_feature_norm = torch.stack([x["tgt_feature_norm"] for x in outputs]).mean()

        self.log_dict(
            {
                "val_loss": loss,
                "val_class_acc": cls_acc,
                "val_src_feature_norm": src_feature_norm,
                "val_tgt_feature_norm": tgt_feature_norm,
            }
        )
