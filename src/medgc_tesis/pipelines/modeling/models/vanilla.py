import json
import logging

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdrift
from sklearn.metrics import f1_score
from tllib.modules.classifier import Classifier
from tllib.utils.metric import ConfusionMatrix, accuracy
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from medgc_tesis.pipelines.modeling.utils import analyze_latent_space

logger = logging.getLogger(__name__)


class SourceOnlyModel(pl.LightningModule):
    def __init__(self, backbone, lr, momentum, weight_decay, lr_gamma, lr_decay):
        super().__init__()
        self.classifier = Classifier(
            backbone.model(),
            num_classes=10,
            pool_layer=backbone.pool_layer(),
            finetune=True,
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
        self.data_transform = backbone.data_transform()

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_gamma = lr_gamma
        self.lr_decay = lr_decay
        self.confusion_matrix = ConfusionMatrix(10)

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
        x_t, _ = batch["tds"]

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

        with open("data/06_models/source_only/metrics.json", "w") as f:
            json.dump(metrics, f)
        with open("data/06_models/source_only/confusion_matrix.txt", "w") as f:
            f.write(self.confusion_matrix.format(list(range(10))))
        df_features.to_parquet("data/06_models/source_only/umap.parquet")

        fig, ax = plt.subplots(figsize=(7, 7))
        df_source = df_features.query("label == 'MNIST'")
        ax.scatter(df_source["0"], df_source["1"], label="MNIST", alpha=0.1)

        df_target = df_features.query("label == 'TDS'")
        ax.scatter(df_target["0"], df_target["1"], label="TDS", alpha=0.1)

        plt.legend()
        fig.savefig("data/06_models/source_only/umap.png")
