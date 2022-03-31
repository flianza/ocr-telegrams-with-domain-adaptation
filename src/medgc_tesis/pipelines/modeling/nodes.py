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
from PIL import Image
from torch.backends import cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from medgc_tesis.pipelines.modeling.models import dann
from medgc_tesis.pipelines.modeling.models.utils import get_model, validate
from medgc_tesis.utils.transforms import get_data_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


logger = logging.getLogger(__name__)


def entrenar_dann(
    params: Dict,
    digitos_mnist_train: ForeverDataIterator,
    digitos_tds_train: ForeverDataIterator,
    digitos_tds_test: ForeverDataIterator,
    digitos_tds_val: ForeverDataIterator,
):
    args = Namespace(**params)
    logger.info(args)
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
    best_acc = 0.0
    best_state = {}
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        logger.info("lr: %f" % lr_scheduler.get_last_lr()[0])

        # train for one epoch
        loss = dann.train_epoch(
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

        # evaluate on validation set
        acc = validate(device, digitos_tds_test.data_loader, classifier)

        # remember best acc and save checkpoint
        if acc > best_acc:
            best_state = classifier.state_dict()
            best_acc = acc
            best_epoch = epoch

        history.append((loss, acc))

    logger.info("best acc: %f" % best_acc)
    logger.info("best epoch: %f" % best_epoch)

    classifier.load_state_dict(best_state)

    acc1 = validate(device, digitos_tds_val.data_loader, classifier)
    logger.info("val acc: %f" % acc1)

    return classifier, pd.DataFrame(history, columns=["loss", "acc"])


def analizar_modelo(
    modelo: Any, digitos_mnist_train: ForeverDataIterator, digitos_tds_train: ForeverDataIterator
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_extractor = nn.Sequential(modelo.backbone, modelo.pool_layer, modelo.bottleneck).to(device)
    source_feature = collect_feature(digitos_mnist_train.data_loader, feature_extractor, device)
    target_feature = collect_feature(digitos_tds_train.data_loader, feature_extractor, device)

    A_distance = a_distance.calculate(source_feature, target_feature, device, training_epochs=4)
    logger.info("A-distance: %f" % A_distance)

    return pd.DataFrame(source_feature.numpy()), pd.DataFrame(target_feature.numpy())


def aplicar_umap(modelo_dann_features_mnist, modelo_dann_features_tds):
    source_feature = modelo_dann_features_mnist.values
    target_feature = modelo_dann_features_tds.values
    features = np.concatenate([source_feature, target_feature], axis=0)

    X_umap = umap.UMAP(random_state=33).fit_transform(features)
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    df = pd.DataFrame(X_umap)
    df["domain"] = domains

    return df


def aplicar_modelo(modelo, dataset_telegramas):
    transform = get_data_transform()

    def predecir_digitos(digitos):
        voto_predicho = ""
        for digito in digitos:
            image = Image.fromarray(np.uint8(np.stack(digito, axis=0)))
            x_transformed = transform(image)
            x_transformed = x_transformed.to(device)
            x_transformed = x_transformed.unsqueeze(0)
            y_pred = modelo(x_transformed)
            y_pred = y_pred.max(1)[1].item()
            voto_predicho += str(y_pred)
        return voto_predicho

    with torch.no_grad():
        dataset_telegramas["voto_predicho"] = dataset_telegramas.digitos.apply(predecir_digitos)
        dataset_telegramas["voto_predicho"] = dataset_telegramas["voto_predicho"].astype(str)

    dataset_telegramas = dataset_telegramas.drop(columns=["digitos"])

    return dataset_telegramas
