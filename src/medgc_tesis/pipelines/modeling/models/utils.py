import argparse
import logging
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tllib.modules.classifier import Classifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.metric import ConfusionMatrix, accuracy
from tllib.vision.models import resnet18, resnet50
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


def validate(device: torch.device, val_loader, model, print_freq=100) -> Tuple[float, str]:
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1], prefix="Test: ")

    # switch to evaluate mode
    model.eval()
    confusion_matrix = ConfusionMatrix(10)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            (acc1,) = accuracy(output, target, topk=(1,))
            confusion_matrix.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

    return top1.avg, confusion_matrix.format(range(10))


def train_epoch(
    device: torch.device,
    train_source_iter: ForeverDataIterator,
    model: Classifier,
    optimizer: SGD,
    lr_scheduler: LambdaLR,
    epoch: int,
    args: argparse.Namespace,
    print_freq=100,
):
    batch_time = AverageMeter("Time", ":3.1f")
    data_time = AverageMeter("Data", ":3.1f")
    losses = AverageMeter("Loss", ":3.2f")
    cls_accs = AverageMeter("Cls Acc", ":3.1f")

    progress = ProgressMeter(
        args.iters_per_epoch, [batch_time, data_time, losses, cls_accs], prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    return losses.avg


class LeNet(nn.Sequential):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.num_classes = num_classes
        self.out_features = 500

    def copy_head(self):
        return nn.Linear(500, self.num_classes)


def get_backbone_model():
    return LeNet()
