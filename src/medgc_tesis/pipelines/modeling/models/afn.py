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
from dalib.adaptation.afn import AdaptiveFeatureNorm, ImageClassifier
from dalib.modules.entropy import entropy
from torch.optim import SGD

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
        num_blocks=args.num_blocks,
        bottleneck_dim=args.bottleneck_dim,
        dropout_p=args.dropout_p,
        # pool_layer=nn.Identity(),
        finetune=True,
    ).to(device)
    adaptive_feature_norm = AdaptiveFeatureNorm(args.delta).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, weight_decay=args.weight_decay)

    # start training
    best_loss = 999.9
    best_state = {}
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):

        # train for one epoch
        loss = train_epoch(
            device,
            train_source_iter=digitos_mnist_train,
            train_target_iter=digitos_tds_train,
            model=classifier,
            adaptive_feature_norm=adaptive_feature_norm,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
        )

        # evaluate on test set
        acc, confusion_matrix = validate(device, digitos_tds_test.data_loader, classifier)
        logger.info("test acc: %f" % acc)
        logger.info(confusion_matrix)

        # remember best acc and save checkpoint
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
    adaptive_feature_norm: AdaptiveFeatureNorm,
    optimizer: SGD,
    epoch: int,
    args: argparse.Namespace,
    print_freq=100,
):
    batch_time = AverageMeter("Time", ":3.1f")
    data_time = AverageMeter("Data", ":3.1f")
    cls_losses = AverageMeter("Cls Loss", ":3.2f")
    norm_losses = AverageMeter("Norm Loss", ":3.2f")
    src_feature_norm = AverageMeter("Source Feature Norm", ":3.2f")
    tgt_feature_norm = AverageMeter("Target Feature Norm", ":3.2f")
    cls_accs = AverageMeter("Cls Acc", ":3.1f")
    tgt_accs = AverageMeter("Tgt Acc", ":3.1f")

    progress = StringProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, norm_losses, src_feature_norm, tgt_feature_norm, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

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
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)

        # classification loss
        cls_loss = F.cross_entropy(y_s, labels_s)
        # norm loss
        norm_loss = adaptive_feature_norm(f_s) + adaptive_feature_norm(f_t)
        loss = cls_loss + norm_loss * args.trade_off_norm

        # using entropy minimization
        if args.trade_off_entropy:
            y_t = F.softmax(y_t, dim=1)
            entropy_loss = entropy(y_t, reduction="mean")
            loss += entropy_loss * args.trade_off_entropy

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update statistics
        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]

        cls_losses.update(cls_loss.item(), x_s.size(0))
        norm_losses.update(norm_loss.item(), x_s.size(0))
        src_feature_norm.update(f_s.norm(p=2, dim=1).mean().item(), x_s.size(0))
        tgt_feature_norm.update(f_t.norm(p=2, dim=1).mean().item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info(progress.display(i))

    return norm_losses.avg
