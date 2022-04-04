import argparse
import logging
import time
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils.data import ForeverDataIterator
from common.utils.meter import AverageMeter
from common.utils.metric import accuracy, binary_accuracy
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.modules.gl import WarmStartGradientLayer
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from medgc_tesis.pipelines.modeling.models.utils import validate
import pandas as pd
from medgc_tesis.pipelines.modeling.models.logging import StringProgressMeter
from dalib.adaptation.adda import ImageClassifier, DomainAdversarialLoss
from dalib.translation.cyclegan.util import set_requires_grad

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
        backbone, 10, bottleneck_dim=args.bottleneck_dim, pool_layer=nn.Identity(), finetune=True
    ).to(device)
    domain_discriminator = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)

    # define loss function
    domain_adv = DomainAdversarialLoss().to(device)
    gradient_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True)

    # define optimizer and lr scheduler
    optimizer = SGD(
        classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True
    )
    optimizer_d = SGD(
        domain_discriminator.get_parameters(),
        args.lr_d,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1.0 + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_d = LambdaLR(optimizer_d, lambda x: args.lr_d * (1.0 + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # start training
    best_acc = 0.0
    best_state = {}
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        logger.info("lr classifier: %f" % lr_scheduler.get_last_lr()[0])
        logger.info("lr discriminator: %f" % lr_scheduler_d.get_last_lr()[0])

        # train for one epoch
        loss = train_epoch(
            device,
            train_source_iter=digitos_mnist_train,
            train_target_iter=digitos_tds_train,
            model=classifier,
            domain_discriminator=domain_discriminator,
            domain_adv=domain_adv,
            gradient_layer=gradient_layer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            optimizer_d=optimizer_d,
            lr_scheduler_d=lr_scheduler_d,
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
    domain_discriminator: DomainDiscriminator,
    domain_adv: DomainAdversarialLoss,
    gradient_layer: nn.Module,
    optimizer: SGD,
    lr_scheduler: LambdaLR,
    optimizer_d: SGD,
    lr_scheduler_d: LambdaLR,
    epoch: int,
    args: argparse.Namespace,
    print_freq=100,
):
    batch_time = AverageMeter("Time", ":5.2f")
    data_time = AverageMeter("Data", ":5.2f")
    losses_s = AverageMeter("Cls Loss", ":6.2f")
    losses_transfer = AverageMeter("Transfer Loss", ":6.2f")
    losses_discriminator = AverageMeter("Discriminator Loss", ":6.2f")
    cls_accs = AverageMeter("Cls Acc", ":3.1f")
    domain_accs = AverageMeter("Domain Acc", ":3.1f")
    progress = StringProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_s, losses_transfer, losses_discriminator, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # Step 1: Train the classifier, freeze the discriminator
        model.train()
        domain_discriminator.eval()
        set_requires_grad(model, True)
        set_requires_grad(domain_discriminator, False)
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        loss_s = F.cross_entropy(y_s, labels_s)

        # adversarial training to fool the discriminator
        d = domain_discriminator(gradient_layer(f))
        d_s, d_t = d.chunk(2, dim=0)
        loss_transfer = 0.5 * (domain_adv(d_s, "target") + domain_adv(d_t, "source"))

        optimizer.zero_grad()
        (loss_s + loss_transfer * args.trade_off).backward()
        optimizer.step()
        lr_scheduler.step()

        # Step 2: Train the discriminator
        model.eval()
        domain_discriminator.train()
        set_requires_grad(model, False)
        set_requires_grad(domain_discriminator, True)
        d = domain_discriminator(f.detach())
        d_s, d_t = d.chunk(2, dim=0)
        loss_discriminator = 0.5 * (domain_adv(d_s, "source") + domain_adv(d_t, "target"))

        optimizer_d.zero_grad()
        loss_discriminator.backward()
        optimizer_d.step()
        lr_scheduler_d.step()

        losses_s.update(loss_s.item(), x_s.size(0))
        losses_transfer.update(loss_transfer.item(), x_s.size(0))
        losses_discriminator.update(loss_discriminator.item(), x_s.size(0))

        cls_acc = accuracy(y_s, labels_s)[0]
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_acc = 0.5 * (binary_accuracy(d_s, torch.ones_like(d_s)) + binary_accuracy(d_t, torch.zeros_like(d_t)))
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info(progress.display(i))

    return losses_discriminator.avg
