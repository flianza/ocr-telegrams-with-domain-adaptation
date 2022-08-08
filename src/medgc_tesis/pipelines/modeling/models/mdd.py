import argparse
import logging
import time
from typing import Any, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tllib.alignment.mdd import (
    ClassificationMarginDisparityDiscrepancy as MarginDisparityDiscrepancy,
)
from tllib.alignment.mdd import ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter
from tllib.utils.metric import accuracy
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
        # pool_layer=nn.Identity(),
        finetune=True,
    ).to(device)
    mdd = MarginDisparityDiscrepancy(args.margin).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(
        classifier.get_parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    lr_scheduler = LambdaLR(
        optimizer,
        lambda x: args.lr * (1.0 + args.lr_gamma * float(x)) ** (-args.lr_decay),
    )

    # start training
    best_loss = 999.9
    best_state = {}
    best_epoch = -1
    history = []
    counter = 0
    for epoch in range(args.epochs):
        logger.info("lr: %f" % lr_scheduler.get_last_lr()[0])

        # train for one epoch
        loss = train_epoch(
            device,
            train_source_iter=digitos_mnist_train,
            train_target_iter=digitos_tds_train,
            model=classifier,
            mdd=mdd,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            args=args,
        )

        # evaluate on test set
        acc, confusion_matrix = validate(device, digitos_tds_test.data_loader, classifier)
        logger.info("test acc: %f" % acc)
        logger.info(confusion_matrix)

        # remember best loss and save checkpoint
        if loss < best_loss:
            best_state = classifier.state_dict()
            best_loss = loss
            best_epoch = epoch
            counter = 0
        else:
            counter += 1
            if counter > args.early_stopping:
                logger.info("stopping at epoch: %f" % epoch)
                break

        history.append((loss, acc))

    logger.info("best loss: %f" % best_loss)
    logger.info("best epoch: %f" % best_epoch)

    classifier.load_state_dict(best_state)

    return classifier, pd.DataFrame(history, columns=["loss", "acc"])


def train_epoch(
    device: torch.device,
    train_source_iter: ForeverDataIterator,
    train_target_iter: ForeverDataIterator,
    model: ImageClassifier,
    mdd: MarginDisparityDiscrepancy,
    optimizer: SGD,
    lr_scheduler: LambdaLR,
    epoch: int,
    args: argparse.Namespace,
    print_freq=100,
):
    batch_time = AverageMeter("Time", ":5.2f")
    data_time = AverageMeter("Data", ":5.2f")
    losses = AverageMeter("Loss", ":6.2f")
    trans_losses = AverageMeter("Trans Loss", ":3.2f")
    cls_accs = AverageMeter("Cls Acc", ":3.1f")
    tgt_accs = AverageMeter("Tgt Acc", ":3.1f")
    progress = StringProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    mdd.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        outputs, outputs_adv = model(x)
        y_s, y_t = outputs.chunk(2, dim=0)
        y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

        # compute losses and acc
        cls_loss = F.cross_entropy(y_s, labels_s)
        # compute margin disparity discrepancy between domains
        # for adversarial classifier, minimize negative mdd is equal to maximize mdd
        transfer_loss = -mdd(y_s, y_s_adv, y_t, y_t_adv)
        loss = cls_loss + transfer_loss * args.trade_off
        model.step()

        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]

        # save losses and acc
        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_t.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))

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
