import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Type

import matplotlib.pyplot as plt
import optuna
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchdrift
from optuna.samplers import RandomSampler
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import f1_score
from tllib.utils.metric import ConfusionMatrix, accuracy
from torch import nn

from medgc_tesis.pipelines.modeling.models.data import DomainAdaptationDataModule
from medgc_tesis.pipelines.modeling.utils import Backbone, analyze_latent_space

logger = logging.getLogger(__name__)


class DomainAdaptationModel(pl.LightningModule):
    def __init__(self, name, classifier):
        super().__init__()
        self.name = name
        self.root_dir = f"data/06_models/{name}"
        self.classifier = classifier
        self.confusion_matrix = ConfusionMatrix(10)

    def forward(self, x):
        return self.classifier(x)

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

        y = self(x_s)

        loss = F.cross_entropy(y, labels_s)

        (acc1,) = accuracy(y, labels_s, topk=(1,))
        f1 = torch.Tensor([f1_score(labels_s.cpu(), y.cpu().argmax(1), average="micro")])
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

        with open(f"{self.root_dir}/metrics.json", "w") as f:
            json.dump(metrics, f)

        with open(f"{self.root_dir}/confusion_matrix.txt", "w") as f:
            f.write(self.confusion_matrix.format(list(range(10))))

        df_features.to_parquet(f"{self.root_dir}/umap.parquet")

        fig, ax = plt.subplots(figsize=(7, 7))
        df_source = df_features.query("label == 'MNIST'")
        ax.scatter(df_source["0"], df_source["1"], label="MNIST", alpha=0.1)

        df_target = df_features.query("label == 'TDS'")
        ax.scatter(df_target["0"], df_target["1"], label="TDS", alpha=0.1)

        plt.legend()
        fig.savefig(f"{self.root_dir}/umap.png")


