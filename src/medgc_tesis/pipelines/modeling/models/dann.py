import argparse
import logging
import time
from typing import Any, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils.data import ForeverDataIterator
from common.utils.meter import AverageMeter
from common.utils.metric import accuracy
from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
from dalib.modules.domain_discriminator import DomainDiscriminator
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from medgc_tesis.pipelines.modeling.models.logging import StringProgressMeter
from medgc_tesis.pipelines.modeling.models.utils import validate

logger = logging.getLogger(__name__)


def train(
    device: torch.device,
    backbone,
    digitos_mnist_train,
    digitos_tds_train,
    digitos_tds_test,
    args,
) -> Tuple[Any, pd.DataFrame]:
    classifier = ImageClassifier(
        backbone,
        num_classes=10,
        bottleneck_dim=args.bottleneck_dim,
        pool_layer=nn.Identity(),
        finetune=True,
    ).to(device)
    domain_discriminator = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(
        classifier.get_parameters() + domain_discriminator.get_parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    lr_scheduler = LambdaLR(
        optimizer,
        lambda x: args.lr * (1.0 + args.lr_gamma * float(x)) ** (-args.lr_decay),
    )

    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discriminator).to(device)

    # start training
    best_acc = 0.0
    best_state = {}
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        logger.info("lr: %f" % lr_scheduler.get_last_lr()[0])

        # train for one epoch
        loss = train_epoch(
            device,
            train_source_iter=digitos_mnist_train,
            train_target_iter=digitos_tds_train,
            model=classifier,
            domain_adv=domain_adv,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            args=args,
        )

        # evaluate on test set
        acc, confusion_matrix = validate(device, digitos_tds_test.data_loader, classifier)
        logger.info("test acc: %f" % acc)
        logger.info(confusion_matrix)

        # remember best acc and save checkpoint
        if acc > best_acc:
            best_state = classifier.state_dict()
            best_acc = acc
            best_epoch = epoch

        history.append((loss, acc))

    logger.info("best acc: %f" % best_acc)
    logger.info("best epoch: %f" % best_epoch)

    classifier.load_state_dict(best_state)

    return classifier, pd.DataFrame(history, columns=["loss", "acc"])


def train_epoch(
    device: torch.device,
    train_source_iter: ForeverDataIterator,
    train_target_iter: ForeverDataIterator,
    model: ImageClassifier,
    domain_adv: DomainAdversarialLoss,
    optimizer: SGD,
    lr_scheduler: LambdaLR,
    epoch: int,
    args: argparse.Namespace,
    print_freq=100,
):
    batch_time = AverageMeter("Time", ":5.2f")
    data_time = AverageMeter("Data", ":5.2f")
    losses = AverageMeter("Loss", ":6.2f")
    cls_accs = AverageMeter("Cls Acc", ":3.1f")
    domain_accs = AverageMeter("Domain Acc", ":3.1f")
    progress = StringProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, _ = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info(progress.display(i))

    return losses.avg
