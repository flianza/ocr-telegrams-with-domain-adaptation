import logging
from argparse import Namespace
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import umap
from common.utils.analysis import a_distance, collect_feature
from common.utils.data import ForeverDataIterator
from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
from dalib.modules.domain_discriminator import DomainDiscriminator
from torch.backends import cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from medgc_tesis.pipelines.modeling.models import dann
from medgc_tesis.pipelines.modeling.models.utils import get_model, validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


logger = logging.getLogger(__name__)


def entrenar_dann(
    params: Dict,
    digitos_mnist_train: ForeverDataIterator,
    digitos_mnist_test: ForeverDataIterator,
    digitos_tds: ForeverDataIterator,
):
    args = Namespace(**params)
    backbone = get_model()
    classifier = ImageClassifier(
        backbone, num_classes=10, bottleneck_dim=args.bottleneck_dim, pool_layer=nn.Identity(), finetune=True
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
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1.0 + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discriminator).to(device)

    # start training
    best_loss = 999
    best_state = {}
    for epoch in range(args.epochs):
        logger.info("lr: %f" % lr_scheduler.get_last_lr()[0])

        # train for one epoch
        loss = dann.train_epoch(
            device,
            train_source_iter=digitos_mnist_train,
            train_target_iter=digitos_tds,
            model=classifier,
            domain_adv=domain_adv,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            args=args,
        )

        # evaluate on validation set
        validate(device, digitos_mnist_test.data_loader, classifier)

        # remember best loss and save checkpoint
        if loss < best_loss:
            best_state = classifier.state_dict()
            best_loss = loss

    logger.info("best loss: %f" % best_loss)

    classifier.load_state_dict(best_state)

    return classifier


def analizar_modelo(
    modelo: Any, digitos_mnist_train: ForeverDataIterator, digitos_tds: ForeverDataIterator
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    feature_extractor = nn.Sequential(modelo.backbone, modelo.pool_layer, modelo.bottleneck).to(device)
    source_feature = collect_feature(digitos_mnist_train.data_loader, feature_extractor, device)
    target_feature = collect_feature(digitos_tds.data_loader, feature_extractor, device)

    # calculate A-distance, which is a measure for distribution discrepancy
    A_distance = a_distance.calculate(source_feature, target_feature, device)
    logger.info("A-distance: %f" % A_distance)

    return pd.DataFrame(source_feature.numpy()), pd.DataFrame(target_feature.numpy())


def aplicar_umap(modelo_dann_features_mnist, modelo_dann_features_tds):

    source_feature = modelo_dann_features_mnist.values
    target_feature = modelo_dann_features_tds.values
    features = np.concatenate([source_feature, target_feature], axis=0)

    X_tsne = umap.UMAP(random_state=33).fit_transform(features)
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    df = pd.DataFrame(X_tsne)
    df["domain"] = domains

    return df
