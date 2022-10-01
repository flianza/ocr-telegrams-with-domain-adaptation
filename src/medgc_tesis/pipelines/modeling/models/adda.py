import logging

import torch
import torch.nn.functional as F
from tllib.alignment.adda import DomainAdversarialLoss, ImageClassifier
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.modules.gl import WarmStartGradientLayer
from tllib.translation.cyclegan.util import set_requires_grad
from tllib.utils.metric import accuracy, binary_accuracy
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from medgc_tesis.pipelines.modeling.models.core import DomainAdaptationModel

logger = logging.getLogger(__name__)


class AddaModel(DomainAdaptationModel):
    def __init__(self, backbone, bottleneck_dim, lr, lr_d, momentum, weight_decay, lr_gamma, lr_decay, trade_off):
        super().__init__(
            da_technique="adda",
            model_name=backbone.name,
            classifier=ImageClassifier(
                backbone.model(),
                num_classes=10,
                bottleneck_dim=bottleneck_dim,
                pool_layer=backbone.pool_layer(),
                finetune=True,
            ),
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

        self.domain_adv = DomainAdversarialLoss()
        self.gradient_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True)

        self.lr = lr
        self.lr_d = lr_d
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_gamma = lr_gamma
        self.lr_decay = lr_decay
        self.trade_off = trade_off

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

        return {
            "class_acc": cls_acc,
            "domain_acc": domain_acc,
            "loss_discriminator": loss_discriminator,
            "loss_transfer": loss_transfer,
            "loss_s": loss_s,
        }

    def training_epoch_end(self, outputs):
        domain_acc = torch.stack([x["domain_acc"] for x in outputs]).mean()
        cls_acc = torch.stack([x["class_acc"] for x in outputs]).mean()
        loss_discriminator = torch.stack([x["loss_discriminator"] for x in outputs]).mean()
        loss_transfer = torch.stack([x["loss_transfer"] for x in outputs]).mean()
        loss_s = torch.stack([x["loss_s"] for x in outputs]).mean()

        self.log_dict(
            {
                "train_domain_acc": domain_acc,
                "train_cls_acc": cls_acc,
                "train_loss_discriminator": loss_discriminator,
                "train_loss_transfer": loss_transfer,
                "train_loss_s": loss_s,
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
            "class_acc": cls_acc,
            "domain_acc": domain_acc,
        }

    def validation_epoch_end(self, outputs) -> None:
        class_acc = torch.stack([x["class_acc"] for x in outputs]).mean()
        domain_acc = torch.stack([x["domain_acc"] for x in outputs]).mean()

        self.log_dict(
            {
                "val_class_acc": class_acc,
                "val_domain_acc": domain_acc,
            }
        )
