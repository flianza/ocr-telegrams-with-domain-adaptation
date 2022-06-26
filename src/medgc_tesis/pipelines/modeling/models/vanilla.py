import logging
from typing import Any, Tuple

import pandas as pd
import torch
import torch.nn as nn
from common.modules.classifier import Classifier
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from medgc_tesis.pipelines.modeling.models.utils import train_epoch, validate

logger = logging.getLogger(__name__)


def train(
    device: torch.device,
    backbone,
    digitos_train,
    digitos_test,
    args,
) -> Tuple[Any, pd.DataFrame]:
    classifier = Classifier(
        backbone,
        num_classes=10,
        # pool_layer=nn.Identity(),
        finetune=True,
    ).to(device)

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
    best_loss = 999.0
    best_state = {}
    best_epoch = -1
    history = []
    counter = 0
    for epoch in range(args.epochs):
        logger.info("lr: %f" % lr_scheduler.get_last_lr()[0])

        # train for one epoch
        loss = train_epoch(
            device,
            train_source_iter=digitos_train,
            model=classifier,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            args=args,
        )

        # evaluate on test set
        acc, confusion_matrix = validate(device, digitos_test.data_loader, classifier)
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
