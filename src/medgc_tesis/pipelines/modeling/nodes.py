import logging

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from pytorch_lightning import seed_everything
from torch.backends import cudnn
from tqdm import tqdm

from medgc_tesis.pipelines.modeling.backbone import LeNetBackbone, ResNet18Backbone
from medgc_tesis.pipelines.modeling.optimizers import (
    adda,
    afn,
    dann,
    mdd,
    source_only,
    target_only,
)

cudnn.benchmark = True
tqdm.pandas()

logger = logging.getLogger(__name__)
seed_everything(48721, workers=True)


def entrenar_lenet_adda():
    adda.setup(backbone=LeNetBackbone)
    return adda.optimize()


def entrenar_resnet_adda():
    adda.setup(backbone=ResNet18Backbone)
    return adda.optimize()


def entrenar_lenet_dann():
    dann.setup(backbone=LeNetBackbone)
    return dann.optimize()


def entrenar_resnet_dann():
    dann.setup(backbone=ResNet18Backbone)
    return dann.optimize()


def entrenar_lenet_mdd():
    mdd.setup(backbone=LeNetBackbone)
    return mdd.optimize()


def entrenar_resnet_mdd():
    mdd.setup(backbone=ResNet18Backbone)
    return mdd.optimize()


def entrenar_lenet_afn():
    afn.setup(backbone=LeNetBackbone)
    return afn.optimize()


def entrenar_resnet_afn():
    afn.setup(backbone=ResNet18Backbone)
    return afn.optimize()


def entrenar_lenet_source_only():
    source_only.setup(backbone=LeNetBackbone)
    return source_only.optimize()


def entrenar_resnet_source_only():
    source_only.setup(backbone=ResNet18Backbone)
    return source_only.optimize()


def entrenar_lenet_target_only():
    target_only.setup(backbone=LeNetBackbone)
    return target_only.optimize()


def entrenar_resnet_target_only():
    target_only.setup(backbone=ResNet18Backbone)
    return target_only.optimize()


def aplicar_modelo(modelo, dataset_telegramas: pd.DataFrame) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = LeNetBackbone().data_transform()
    modelo = modelo.to("cuda")
    modelo.eval()

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
