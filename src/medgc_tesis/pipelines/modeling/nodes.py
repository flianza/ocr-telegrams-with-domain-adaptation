import logging
from argparse import Namespace
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import umap
from matplotlib import pyplot as plt
from PIL import Image
from tllib.utils.analysis import a_distance, collect_feature
from tllib.utils.data import ForeverDataIterator
from torch.backends import cudnn
from tqdm import tqdm

from medgc_tesis.pipelines.modeling.models import adda, afn, dann, mdd, vanilla
from medgc_tesis.pipelines.modeling.models.utils import get_backbone_model, validate
from medgc_tesis.utils.transforms import get_data_transform

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cudnn.benchmark = True
tqdm.pandas()

logger = logging.getLogger(__name__)


def entrenar_vanilla(
    params: Dict,
    digitos_train: ForeverDataIterator,
    digitos_test: ForeverDataIterator,
    digitos_val: ForeverDataIterator,
) -> Tuple[Any, pd.DataFrame, str]:
    args = Namespace(**params)
    logger.info(args)

    backbone = get_backbone_model()
    classifier, history = vanilla.train(device, backbone, digitos_train, digitos_test, args)

    _, confusion_matrix_train = validate(device, digitos_train.data_loader, classifier)
    _, confusion_matrix_test = validate(device, digitos_test.data_loader, classifier)
    acc, confusion_matrix_val = validate(device, digitos_val.data_loader, classifier)
    logger.info("val acc: %f" % acc)
    logger.info(confusion_matrix_val)

    metrics = f"TRAIN\n{confusion_matrix_train}\n\nTEST\n{confusion_matrix_test}\n\nVAL\n{confusion_matrix_val}"

    return classifier, history, metrics


def entrenar_dann(
    params: Dict,
    digitos_mnist_train: ForeverDataIterator,
    digitos_tds_train: ForeverDataIterator,
    digitos_tds_test: ForeverDataIterator,
    digitos_tds_val: ForeverDataIterator,
) -> Tuple[Any, pd.DataFrame, str]:
    args = Namespace(**params)
    logger.info(args)

    backbone = get_backbone_model()
    classifier, history = dann.train(device, backbone, digitos_mnist_train, digitos_tds_train, digitos_tds_test, args)

    _, confusion_matrix_train = validate(device, digitos_tds_train.data_loader, classifier)
    _, confusion_matrix_test = validate(device, digitos_tds_test.data_loader, classifier)
    acc, confusion_matrix_val = validate(device, digitos_tds_val.data_loader, classifier)
    logger.info("val acc: %f" % acc)
    logger.info(confusion_matrix_val)

    metrics = f"TRAIN\n{confusion_matrix_train}\n\nTEST\n{confusion_matrix_test}\n\nVAL\n{confusion_matrix_val}"

    return classifier, history, metrics


def entrenar_afn(
    params: Dict,
    digitos_mnist_train: ForeverDataIterator,
    digitos_tds_train: ForeverDataIterator,
    digitos_tds_test: ForeverDataIterator,
    digitos_tds_val: ForeverDataIterator,
) -> Tuple[Any, pd.DataFrame, str]:
    args = Namespace(**params)
    logger.info(args)

    backbone = get_backbone_model()
    classifier, history = afn.train(device, backbone, digitos_mnist_train, digitos_tds_train, digitos_tds_test, args)

    _, confusion_matrix_train = validate(device, digitos_tds_train.data_loader, classifier)
    _, confusion_matrix_test = validate(device, digitos_tds_test.data_loader, classifier)
    acc, confusion_matrix_val = validate(device, digitos_tds_val.data_loader, classifier)
    logger.info("val acc: %f" % acc)
    logger.info(confusion_matrix_val)

    metrics = f"TRAIN\n{confusion_matrix_train}\n\nTEST\n{confusion_matrix_test}\n\nVAL\n{confusion_matrix_val}"

    return classifier, history, metrics


def entrenar_adda(
    params: Dict,
    digitos_mnist_train: ForeverDataIterator,
    digitos_tds_train: ForeverDataIterator,
    digitos_tds_test: ForeverDataIterator,
    digitos_tds_val: ForeverDataIterator,
) -> Tuple[Any, pd.DataFrame, str]:
    args = Namespace(**params)
    logger.info(args)

    backbone = get_backbone_model()
    classifier, history = adda.train(device, backbone, digitos_mnist_train, digitos_tds_train, digitos_tds_test, args)

    _, confusion_matrix_train = validate(device, digitos_tds_train.data_loader, classifier)
    _, confusion_matrix_test = validate(device, digitos_tds_test.data_loader, classifier)
    acc, confusion_matrix_val = validate(device, digitos_tds_val.data_loader, classifier)
    logger.info("val acc: %f" % acc)
    logger.info(confusion_matrix_val)

    metrics = f"TRAIN\n{confusion_matrix_train}\n\nTEST\n{confusion_matrix_test}\n\nVAL\n{confusion_matrix_val}"

    return classifier, history, metrics


def entrenar_mdd(
    params: Dict,
    digitos_mnist_train: ForeverDataIterator,
    digitos_tds_train: ForeverDataIterator,
    digitos_tds_test: ForeverDataIterator,
    digitos_tds_val: ForeverDataIterator,
) -> Tuple[Any, pd.DataFrame, str]:
    args = Namespace(**params)
    logger.info(args)

    backbone = get_backbone_model()
    classifier, history = mdd.train(device, backbone, digitos_mnist_train, digitos_tds_train, digitos_tds_test, args)

    _, confusion_matrix_train = validate(device, digitos_tds_train.data_loader, classifier)
    _, confusion_matrix_test = validate(device, digitos_tds_test.data_loader, classifier)
    acc, confusion_matrix_val = validate(device, digitos_tds_val.data_loader, classifier)
    logger.info("val acc: %f" % acc)
    logger.info(confusion_matrix_val)

    metrics = f"TRAIN\n{confusion_matrix_train}\n\nTEST\n{confusion_matrix_test}\n\nVAL\n{confusion_matrix_val}"

    return classifier, history, metrics


def extraer_features(
    modelo: Any,
    digitos_source: ForeverDataIterator,
    digitos_target: ForeverDataIterator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if hasattr(modelo, "pool_layer"):
        feature_extractor = nn.Sequential(modelo.backbone, modelo.pool_layer, modelo.bottleneck).to(device)
    else:
        feature_extractor = nn.Sequential(modelo.backbone, modelo.bottleneck).to(device)
    source_feature = collect_feature(digitos_source.data_loader, feature_extractor, device)
    target_feature = collect_feature(digitos_target.data_loader, feature_extractor, device)

    # A_distance = a_distance.calculate(source_feature, target_feature, device, training_epochs=4)
    # logger.info("A-distance: %f" % A_distance)

    df_source_feature = pd.DataFrame(source_feature.numpy())
    df_source_feature.columns = df_source_feature.columns.astype(str)
    df_target_feature = pd.DataFrame(target_feature.numpy())
    df_target_feature.columns = df_target_feature.columns.astype(str)

    return df_source_feature, df_target_feature


def aplicar_umap(features_modelo_mnist, features_modelo_tds):
    source_feature = features_modelo_mnist.values
    target_feature = features_modelo_tds.values
    features = np.concatenate([source_feature, target_feature], axis=0)

    X_umap = umap.UMAP(random_state=33).fit_transform(features)
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    df = pd.DataFrame(X_umap)
    df["domain"] = domains
    df.columns = df.columns.astype(str)

    return df


def graficar_umap(df_umap: pd.DataFrame) -> plt.figure:
    df_source = df_umap.query("domain == 1.0")
    df_target = df_umap.query("domain == 0.0").sample(n=df_source.shape[0])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df_target["0"], df_target["1"], label="target", alpha=0.1)
    ax.scatter(df_source["0"], df_source["1"], label="source", alpha=0.1)
    plt.legend()
    return fig


def aplicar_modelo(modelo, dataset_telegramas: pd.DataFrame) -> pd.DataFrame:
    transform = get_data_transform()

    def predecir_digitos(digitos):
        xs = []
        for digito in digitos:
            image = Image.fromarray(np.uint8(np.stack(digito, axis=0)))
            x_transformed = transform(image)
            x_transformed = x_transformed.to(device)
            x_transformed = x_transformed.unsqueeze(0)
            xs.append(x_transformed)
        xs = torch.cat(xs)
        y_hat = modelo(xs)
        y_pred = y_hat.max(1)[1].cpu().numpy()
        return "".join(str(x) for x in y_pred)

    with torch.no_grad():
        dataset_telegramas["voto_predicho"] = dataset_telegramas.digitos.progress_apply(predecir_digitos)

    dataset_telegramas["voto_predicho"] = dataset_telegramas["voto_predicho"].astype(str)
    dataset_telegramas = dataset_telegramas.drop(columns=["digitos"])

    def calcular_iou(x):
        voto_real = str(x["votos"])
        voto_predicho = str(int(x["voto_predicho"]))
        intersection = len(set(voto_predicho) & set(voto_real))
        union = len(set(voto_predicho) | set(voto_real))
        return intersection / union

    dataset_telegramas["iou"] = dataset_telegramas.apply(calcular_iou, axis=1)

    logger.info(f"IOU promedio: {dataset_telegramas['iou'].mean()}")

    return dataset_telegramas
