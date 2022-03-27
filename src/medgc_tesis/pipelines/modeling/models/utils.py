import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.metric import ConfusionMatrix, accuracy


def validate(device: torch.device, val_loader, model, print_freq=100) -> float:
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

        print(f" * Acc@1 {top1.avg:.3f}")
        if confusion_matrix:
            print(confusion_matrix.format(range(10)))

    return top1.avg


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


def get_model():
    return LeNet()
