import json
import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchdrift
from sklearn.metrics import f1_score
from tllib.alignment.adda import DomainAdversarialLoss, ImageClassifier
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.modules.gl import WarmStartGradientLayer
from tllib.translation.cyclegan.util import set_requires_grad
from tllib.utils.metric import ConfusionMatrix, accuracy, binary_accuracy
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from medgc_tesis.pipelines.modeling.utils import analyze_latent_space

logger = logging.getLogger(__name__)


class AddaModel(pl.LightningModule):
    def __init__(self, backbone, bottleneck_dim, lr, lr_d, momentum, weight_decay, lr_gamma, lr_decay, trade_off):
        super().__init__()
        self.classifier = ImageClassifier(
            backbone.model(),
            num_classes=10,
            bottleneck_dim=bottleneck_dim,
            pool_layer=backbone.pool_layer(),
            finetune=True,
        )
        self.save_hyperparameters(
            {
                "bottleneck_dim": bottleneck_dim,
                "lr": lr,
                "lr_d": lr_d,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "lr_gamma": lr_gamma,
                "lr_decay": lr_decay,
                "trade_off": trade_off,
            }
        )
        self.domain_discriminator = DomainDiscriminator(in_feature=self.classifier.features_dim, hidden_size=1024)
        self.data_transform = backbone.data_transform()

        self.domain_adv = DomainAdversarialLoss()
        self.gradient_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True)

        self.lr = lr
        self.lr_d = lr_d
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_gamma = lr_gamma
        self.lr_decay = lr_decay
        self.trade_off = trade_off
        self.confusion_matrix = ConfusionMatrix(10)

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = SGD(
            self.classifier.get_parameters(),
            self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )
        optimizer_d = SGD(
            self.domain_discriminator.get_parameters(),
            self.lr_d,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )
        lr_scheduler = LambdaLR(optimizer, lambda x: self.lr * (1.0 + self.lr_gamma * float(x)) ** (-self.lr_decay))
        lr_scheduler_d = LambdaLR(
            optimizer_d, lambda x: self.lr_d * (1.0 + self.lr_gamma * float(x)) ** (-self.lr_decay)
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}, {
            "optimizer": optimizer_d,
            "lr_scheduler": lr_scheduler_d,
        }

    def training_step(self, batch, batch_idx):
        x_s, labels_s = batch["mnist"]
        x_t, _ = batch["tds"]

        optimizer, optimizer_d = self.optimizers()

        # Step 1: Train the classifier, freeze the discriminator
        self.classifier.train()
        self.domain_discriminator.eval()
        set_requires_grad(self.classifier, True)
        set_requires_grad(self.domain_discriminator, False)
        x = torch.cat((x_s, x_t), dim=0)
        y, f = self.classifier(x)
        y_s, _ = y.chunk(2, dim=0)
        loss_s = F.cross_entropy(y_s, labels_s)

        # adversarial training to fool the discriminator
        d = self.domain_discriminator(self.gradient_layer(f))
        d_s, d_t = d.chunk(2, dim=0)
        loss_transfer = 0.5 * (self.domain_adv(d_s, "target") + self.domain_adv(d_t, "source"))

        optimizer.zero_grad()
        (loss_s + loss_transfer * self.trade_off).backward()
        optimizer.step()

        # Step 2: Train the discriminator
        self.classifier.eval()
        self.domain_discriminator.train()
        set_requires_grad(self.classifier, False)
        set_requires_grad(self.domain_discriminator, True)
        d = self.domain_discriminator(f.detach())
        d_s, d_t = d.chunk(2, dim=0)
        loss_discriminator = 0.5 * (self.domain_adv(d_s, "source") + self.domain_adv(d_t, "target"))

        optimizer_d.zero_grad()
        loss_discriminator.backward()
        optimizer_d.step()

        cls_acc = accuracy(y_s, labels_s)[0]
        domain_acc = 0.5 * (binary_accuracy(d_s, torch.ones_like(d_s)) + binary_accuracy(d_t, torch.zeros_like(d_t)))

        metrics = {
            "train_class_acc_step": cls_acc,
            "train_domain_acc_step": domain_acc,
            "train_loss_discriminator_step": loss_discriminator,
            "train_loss_transfer_step": loss_transfer,
            "train_loss_s_step": loss_s,
        }

        return metrics

    def training_epoch_end(self, outputs):
        avg_domain_acc = torch.stack([x["train_domain_acc_step"] for x in outputs]).mean()
        avg_cls_acc = torch.stack([x["train_class_acc_step"] for x in outputs]).mean()
        avg_loss_discriminator = torch.stack([x["train_loss_discriminator_step"] for x in outputs]).mean()
        avg_loss_transfer = torch.stack([x["train_loss_transfer_step"] for x in outputs]).mean()
        avg_loss_s = torch.stack([x["train_loss_s_step"] for x in outputs]).mean()

        self.log_dict(
            {
                "train_domain_acc": avg_domain_acc,
                "train_cls_acc": avg_cls_acc,
                "train_loss_discriminator": avg_loss_discriminator,
                "train_loss_transfer": avg_loss_transfer,
                "train_loss_s": avg_loss_s,
            }
        )

    def validation_step(self, batch, batch_idx):
        x_s, labels_s = batch["mnist"]
        x_t, _ = batch["tds"]

        self.classifier.train()
        self.domain_discriminator.eval()

        x = torch.cat((x_s, x_t), dim=0)
        y, f = self.classifier(x)
        d = self.domain_discriminator(self.gradient_layer(f))

        y_s, _ = y.chunk(2, dim=0)
        d_s, d_t = d.chunk(2, dim=0)

        cls_acc = accuracy(y_s, labels_s)[0]
        domain_acc = 0.5 * (binary_accuracy(d_s, torch.ones_like(d_s)) + binary_accuracy(d_t, torch.zeros_like(d_t)))

        return {
            "val_class_acc_step": cls_acc,
            "val_domain_acc_step": domain_acc,
        }

    def validation_epoch_end(self, outputs) -> None:
        avg_class_acc = torch.stack([x["val_class_acc_step"] for x in outputs]).mean()
        avg_domain_acc = torch.stack([x["val_domain_acc_step"] for x in outputs]).mean()

        self.log_dict(
            {
                "val_class_acc": avg_class_acc,
                "val_domain_acc": avg_domain_acc,
            }
        )

    def on_test_epoch_start(self) -> None:
        if hasattr(self.classifier, "pool_layer"):
            self.feature_extractor = nn.Sequential(
                self.classifier.backbone, self.classifier.pool_layer, self.classifier.bottleneck
            ).to(self.device)
        else:
            self.feature_extractor = nn.Sequential(self.classifier.backbone, self.classifier.bottleneck).to(self.device)

        test_mnist_dataloader, _ = self.trainer.datamodule.test_dataloader().loaders.values()
        self.drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
        torchdrift.utils.fit(test_mnist_dataloader, self.feature_extractor, self.drift_detector)

    def test_step(self, batch, batch_idx):
        x_s, labels_s = batch["mnist"]
        x_t, _ = batch["tds"]

        y = self.classifier(x_s)

        loss = F.cross_entropy(y, labels_s)

        (acc1,) = accuracy(y, labels_s, topk=(1,))
        f1 = f1_score(labels_s.cpu(), y.cpu().argmax(1), average="micro")
        self.confusion_matrix.update(labels_s.cpu(), y.cpu().argmax(1))

        feature = self.feature_extractor(x_t)
        score = self.drift_detector(feature)
        pvalue = self.drift_detector.compute_p_value(feature)

        return {
            "test_acc_step": acc1,
            "test_loss_step": loss,
            "test_f1_step": f1,
            "test_drift_feature_step": feature,
            "test_drift_score_step": score,
            "test_drift_pvalue_step": pvalue,
        }

    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x["test_acc_step"] for x in outputs]).mean().item()
        avg_loss = torch.stack([x["test_loss_step"] for x in outputs]).mean().item()
        avg_f1 = torch.stack([x["test_f1_step"] for x in outputs]).mean().item()
        avg_drift_score = torch.stack([x["test_drift_score_step"] for x in outputs]).mean().item()
        avg_drift_pvalue = torch.stack([x["test_drift_pvalue_step"] for x in outputs]).mean().item()
        tds_features = torch.cat([x["test_drift_feature_step"] for x in outputs], dim=0)

        A_distance, df_features = analyze_latent_space(self.drift_detector.base_outputs.cpu(), tds_features.cpu())

        metrics = {
            "test_acc": avg_acc,
            "test_loss": avg_loss,
            "test_f1": avg_f1,
            "test_drift_score": avg_drift_score,
            "test_drift_pvalue": avg_drift_pvalue,
            "test_a_distance": A_distance,
        }
        self.log_dict(metrics)

        with open("data/06_models/adda/metrics.json", "w") as f:
            json.dump(metrics, f)
        with open("data/06_models/adda/confusion_matrix.txt", "w") as f:
            f.write(self.confusion_matrix.format(list(range(10))))
        df_features.to_parquet("data/06_models/adda/umap.parquet")

        fig, ax = plt.subplots(figsize=(7, 7))
        df_source = df_features.query("label == 'MNIST'")
        ax.scatter(df_source["0"], df_source["1"], label="MNIST", alpha=0.1)

        df_target = df_features.query("label == 'TDS'")
        ax.scatter(df_target["0"], df_target["1"], label="TDS", alpha=0.1)

        plt.legend()
        fig.savefig("data/06_models/adda/umap.png")
