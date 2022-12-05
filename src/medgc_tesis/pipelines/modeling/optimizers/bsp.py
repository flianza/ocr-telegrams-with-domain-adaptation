import json
from typing import Type

import optuna
from optuna.samplers import RandomSampler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from medgc_tesis.pipelines.modeling.backbone import Backbone
from medgc_tesis.pipelines.modeling.models.bsp import BspModel
from medgc_tesis.pipelines.modeling.models.data import DomainAdaptationDataModule
from medgc_tesis.pipelines.modeling.optimizers import defaults

backbone_class: Type[Backbone] = None
experiment_name = ""
da_technique = "bsp"


def setup(backbone: Type[Backbone]):
    global experiment_name, backbone_class
    backbone_class = backbone
    experiment_name = f"{da_technique}_{backbone_class.name}"


def optimize():
    study = optuna.create_study(
        directions=["minimize", "maximize"],
        study_name=experiment_name,
        sampler=RandomSampler(42),
    )
    study.optimize(_suggest_bsp, n_trials=defaults.OPTUNA_N_TRIALS)

    best_params = study.best_trials[0].params

    best_model, _ = _fit_bsp(best_params, test=True)

    return best_model


def _suggest_bsp(trial: optuna.Trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 0.1),
        "trade_off": trial.suggest_float("trade_off", 0.5, 2),
        "trade_off_bsp": trial.suggest_float("trade_off_bsp", 1e-5, 0.1),
    }
    _, val_metrics = _fit_bsp(params, test=False)

    return val_metrics["val_loss"], val_metrics["val_class_acc"]


def _fit_bsp(params, test):
    backbone = backbone_class()

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

    model = BspModel(backbone=backbone, **params)

    trainer = Trainer(
        accelerator="gpu",
        gpus=0,
        max_epochs=defaults.TRAIN_MAX_EPOCHS,
        callbacks=[
            EarlyStopping(monitor="val_domain_acc", mode="min", patience=3),
            ModelCheckpoint(
                monitor="val_domain_acc",
                filename=experiment_name + "-{epoch:02d}-{val_domain_acc:.4f}",
            ),
        ],
        deterministic=True,
        logger=TensorBoardLogger(f"tb_logs/{da_technique}", name=backbone_class.name),
    )

    trainer.fit(model, datamodule=dm)

    val_metrics = trainer.validate(model, datamodule=dm)

    if test:
        with open(f"data/06_models/{da_technique}/{backbone.name}/best_params.json", "w") as f:
            json.dump(params, f)
        trainer.test(model, datamodule=dm)

    return model.classifier, val_metrics[0]
