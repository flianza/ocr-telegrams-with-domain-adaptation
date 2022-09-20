import json
import logging

import numpy as np
import optuna
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from optuna.samplers import RandomSampler
from PIL import Image
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.backends import cudnn
from tqdm import tqdm

from medgc_tesis.pipelines.modeling.models import adda, afn, dann, mdd, vanilla
from medgc_tesis.pipelines.modeling.models.data import DomainAdaptationDataModule
from medgc_tesis.pipelines.modeling.utils import LeNetBackbone
from medgc_tesis.utils.transforms import get_data_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
tqdm.pandas()

logger = logging.getLogger(__name__)
seed_everything(48721, workers=True)

# region ADDA
def entrenar_adda():
    model_name = "adda"

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        study_name=model_name,
        sampler=RandomSampler(42),
    )
    study.optimize(_suggest_adda, n_trials=15)

    best_params = study.best_trials[0].params

    best_model, _ = _fit_adda(best_params, test=True)

    return best_model


def _suggest_adda(trial: optuna.Trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 0.1),
        "lr_d": trial.suggest_float("lr_d", 1e-4, 0.1),
        "trade_off": trial.suggest_float("trade_off", 0.5, 2),
    }
    _, val_metrics = _fit_adda(params, test=False)

    return val_metrics["val_domain_acc"], val_metrics["val_class_acc"]


def _fit_adda(params, test):
    model_name = "adda"
    backbone = LeNetBackbone()

    dm = DomainAdaptationDataModule(transform=backbone.data_transform())

    params.update(
        {
            "lr_gamma": 0.001,
            "lr_decay": 0.25,
            "momentum": 0.9,
            "weight_decay": 0.001,
            "bottleneck_dim": 256,
        }
    )

    model = adda.AddaModel(backbone=backbone, **params)

    trainer = Trainer(
        accelerator="gpu",
        gpus=0,
        max_epochs=10,
        callbacks=[
            EarlyStopping(monitor="val_domain_acc", mode="min", patience=3),
            ModelCheckpoint(
                monitor="val_domain_acc",
                filename=model_name + "-{epoch:02d}-{val_domain_acc:.4f}",
            ),
        ],
        deterministic=True,
        logger=TensorBoardLogger("tb_logs", name=model_name),
    )

    trainer.fit(model, datamodule=dm)

    val_metrics = trainer.validate(model, datamodule=dm)

    if test:
        with open(f"data/06_models/{model_name}/best_params.json", "w") as f:
            json.dump(params, f)
        trainer.test(model, datamodule=dm)

    return model.classifier, val_metrics[0]


# endregion

# region DANN
def entrenar_dann():
    model_name = "dann"

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        study_name=model_name,
        sampler=RandomSampler(42),
    )
    study.optimize(_suggest_dann, n_trials=15)

    best_params = study.best_trials[0].params

    best_model, _ = _fit_dann(best_params, test=True)

    return best_model


def _suggest_dann(trial: optuna.Trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 0.1),
        "trade_off": trial.suggest_float("trade_off", 0.5, 2),
    }
    _, val_metrics = _fit_dann(params, test=False)

    return val_metrics["val_domain_acc"], val_metrics["val_class_acc"]


def _fit_dann(params, test):
    model_name = "dann"
    backbone = LeNetBackbone()

    dm = DomainAdaptationDataModule(transform=backbone.data_transform())

    params.update(
        {
            "lr_gamma": 0.001,
            "lr_decay": 0.25,
            "momentum": 0.9,
            "weight_decay": 0.001,
            "bottleneck_dim": 256,
        }
    )

    model = dann.DannModel(backbone=backbone, **params)

    trainer = Trainer(
        accelerator="gpu",
        gpus=0,
        max_epochs=10,
        callbacks=[
            EarlyStopping(monitor="val_domain_acc", mode="min", patience=3),
            ModelCheckpoint(
                monitor="val_domain_acc",
                filename=model_name + "-{epoch:02d}-{val_domain_acc:.4f}",
            ),
        ],
        deterministic=True,
        logger=TensorBoardLogger("tb_logs", name=model_name),
    )

    trainer.fit(model, datamodule=dm)

    val_metrics = trainer.validate(model, datamodule=dm)

    if test:
        with open(f"data/06_models/{model_name}/best_params.json", "w") as f:
            json.dump(params, f)
        trainer.test(model, datamodule=dm)

    return model.classifier, val_metrics[0]


# endregion

# region MDD
def entrenar_mdd():
    model_name = "mdd"

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        study_name=model_name,
        sampler=RandomSampler(42),
    )
    study.optimize(_suggest_mdd, n_trials=15)

    best_params = study.best_trials[0].params

    best_model, _ = _fit_mdd(best_params, test=True)

    return best_model


def _suggest_mdd(trial: optuna.Trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 0.1),
        "trade_off": trial.suggest_float("trade_off", 0.5, 2),
        "margin": trial.suggest_float("margin", 1, 10),
    }
    _, val_metrics = _fit_mdd(params, test=False)

    return val_metrics["val_loss"], val_metrics["val_class_acc"]


def _fit_mdd(params, test):
    model_name = "mdd"
    backbone = LeNetBackbone()

    dm = DomainAdaptationDataModule(transform=backbone.data_transform())

    params.update(
        {
            "lr_gamma": 0.001,
            "lr_decay": 0.25,
            "momentum": 0.9,
            "weight_decay": 0.001,
            "bottleneck_dim": 256,
        }
    )

    model = mdd.MddModel(backbone=backbone, **params)

    trainer = Trainer(
        accelerator="gpu",
        gpus=0,
        max_epochs=10,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
            ModelCheckpoint(
                monitor="val_loss",
                filename=model_name + "-{epoch:02d}-{val_loss:.4f}",
            ),
        ],
        deterministic=True,
        logger=TensorBoardLogger("tb_logs", name=model_name),
    )

    trainer.fit(model, datamodule=dm)

    val_metrics = trainer.validate(model, datamodule=dm)

    if test:
        with open(f"data/06_models/{model_name}/best_params.json", "w") as f:
            json.dump(params, f)
        trainer.test(model, datamodule=dm)

    return model.classifier, val_metrics[0]


# endregion

# region AFN
def entrenar_afn():
    model_name = "afn"

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        study_name=model_name,
        sampler=RandomSampler(42),
    )
    study.optimize(_suggest_afn, n_trials=15)

    best_params = study.best_trials[0].params

    del best_params["enable_trade_off_entropy"]
    best_model, _ = _fit_afn(best_params, test=True)

    return best_model


def _suggest_afn(trial: optuna.Trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 0.1),
        "delta": trial.suggest_uniform("delta", 0.01, 5),
        "num_blocks": trial.suggest_int("num_blocks", 1, 4),
        "dropout_p": trial.suggest_float("dropout_p", 0.3, 0.7),
        "trade_off_norm": trial.suggest_uniform("trade_off_norm", 0.001, 0.1),
    }

    enable_trade_off_entropy = trial.suggest_categorical("enable_trade_off_entropy", [True, False])
    if enable_trade_off_entropy:
        params.update(
            {
                "trade_off_entropy": trial.suggest_uniform("trade_off_entropy", 0.001, 0.1),
            }
        )

    _, val_metrics = _fit_afn(params, test=False)

    return val_metrics["val_loss"], val_metrics["val_class_acc"]


def _fit_afn(params, test):
    model_name = "afn"
    backbone = LeNetBackbone()

    dm = DomainAdaptationDataModule(transform=backbone.data_transform())

    params.update(
        {
            "weight_decay": 0.001,
            "bottleneck_dim": 256,
        }
    )

    model = afn.AfnModel(backbone=backbone, **params)

    trainer = Trainer(
        accelerator="gpu",
        gpus=0,
        max_epochs=10,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
            ModelCheckpoint(
                monitor="val_loss",
                filename=model_name + "-{epoch:02d}-{val_loss:.4f}",
            ),
        ],
        deterministic=True,
        logger=TensorBoardLogger("tb_logs", name=model_name),
    )

    trainer.fit(model, datamodule=dm)

    val_metrics = trainer.validate(model, datamodule=dm)

    if test:
        with open(f"data/06_models/{model_name}/best_params.json", "w") as f:
            json.dump(params, f)
        trainer.test(model, datamodule=dm)

    return model.classifier, val_metrics[0]


# endregion

# region source_only
def entrenar_source_only():
    model_name = "source_only"

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        study_name=model_name,
        sampler=RandomSampler(42),
    )
    study.optimize(_suggest_source_only, n_trials=15)

    best_params = study.best_trials[0].params

    best_model, _ = _fit_source_only(best_params, test=True)

    return best_model


def _suggest_source_only(trial: optuna.Trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 0.1),
    }
    _, val_metrics = _fit_source_only(params, test=False)

    return val_metrics["val_loss"], val_metrics["val_class_acc"]


def _fit_source_only(params, test):
    model_name = "source_only"
    backbone = LeNetBackbone()

    dm = DomainAdaptationDataModule(transform=backbone.data_transform())

    params.update(
        {
            "lr_gamma": 0.001,
            "lr_decay": 0.25,
            "momentum": 0.9,
            "weight_decay": 0.001,
        }
    )

    model = vanilla.SourceOnlyModel(backbone=backbone, **params)

    trainer = Trainer(
        accelerator="gpu",
        gpus=0,
        max_epochs=10,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
            ModelCheckpoint(
                monitor="val_loss",
                filename=model_name + "-{epoch:02d}-{val_loss:.4f}",
            ),
        ],
        deterministic=True,
        logger=TensorBoardLogger("tb_logs", name=model_name),
    )

    trainer.fit(model, datamodule=dm)

    val_metrics = trainer.validate(model, datamodule=dm)

    if test:
        with open(f"data/06_models/{model_name}/best_params.json", "w") as f:
            json.dump(params, f)
        trainer.test(model, datamodule=dm)

    return model.classifier, val_metrics[0]


# endregion

# region target_only
def entrenar_target_only():
    model_name = "target_only"

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        study_name=model_name,
        sampler=RandomSampler(42),
    )
    study.optimize(_suggest_target_only, n_trials=15)

    best_params = study.best_trials[0].params

    best_model, _ = _fit_target_only(best_params, test=True)

    return best_model


def _suggest_target_only(trial: optuna.Trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 0.1),
    }
    _, val_metrics = _fit_target_only(params, test=False)

    return val_metrics["val_loss"], val_metrics["val_class_acc"]


def _fit_target_only(params, test):
    model_name = "target_only"
    backbone = LeNetBackbone()

    dm = DomainAdaptationDataModule(transform=backbone.data_transform())

    params.update(
        {
            "lr_gamma": 0.001,
            "lr_decay": 0.25,
            "momentum": 0.9,
            "weight_decay": 0.001,
        }
    )

    model = vanilla.TargetOnlyModel(backbone=backbone, **params)

    trainer = Trainer(
        accelerator="gpu",
        gpus=0,
        max_epochs=10,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
            ModelCheckpoint(
                monitor="val_loss",
                filename=model_name + "-{epoch:02d}-{val_loss:.4f}",
            ),
        ],
        deterministic=True,
        logger=TensorBoardLogger("tb_logs", name=model_name),
    )

    trainer.fit(model, datamodule=dm)

    val_metrics = trainer.validate(model, datamodule=dm)

    if test:
        with open(f"data/06_models/{model_name}/best_params.json", "w") as f:
            json.dump(params, f)
        trainer.test(model, datamodule=dm)

    return model.classifier, val_metrics[0]


# endregion


def aplicar_modelo(modelo, dataset_telegramas: pd.DataFrame) -> pd.DataFrame:
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
