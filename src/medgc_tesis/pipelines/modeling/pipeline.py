from typing import Callable

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from medgc_tesis.pipelines.modeling.nodes import (
    aplicar_modelo,
    entrenar_lenet_adda,
    entrenar_lenet_afn,
    entrenar_lenet_dann,
    entrenar_lenet_mdd,
    entrenar_lenet_source_only,
    entrenar_lenet_target_only,
    entrenar_resnet_adda,
    entrenar_resnet_afn,
    entrenar_resnet_dann,
    entrenar_resnet_mdd,
    entrenar_resnet_source_only,
    entrenar_resnet_target_only,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # ADDA
            node(
                entrenar_lenet_adda,
                inputs=None,
                outputs="modelo_lenet_adda",
                name="entrenar_lenet_adda",
            ),
            node(
                entrenar_resnet_adda,
                inputs=None,
                outputs="modelo_resnet_adda",
                name="entrenar_resnet_adda",
            ),
            # SOURCE_ONLY
            node(
                entrenar_lenet_source_only,
                inputs=None,
                outputs="modelo_lenet_source_only",
                name="entrenar_lenet_source_only",
            ),
            node(
                entrenar_resnet_source_only,
                inputs=None,
                outputs="modelo_resnet_source_only",
                name="entrenar_resnet_source_only",
            ),
            # TARGET_ONLY
            node(
                entrenar_lenet_target_only,
                inputs=None,
                outputs="modelo_lenet_target_only",
                name="entrenar_lenet_target_only",
            ),
            node(
                entrenar_resnet_target_only,
                inputs=None,
                outputs="modelo_resnet_target_only",
                name="entrenar_resnet_target_only",
            ),
            # DANN
            node(
                entrenar_lenet_dann,
                inputs=None,
                outputs="modelo_lenet_dann",
                name="entrenar_lenet_dann",
            ),
            node(
                entrenar_resnet_dann,
                inputs=None,
                outputs="modelo_resnet_dann",
                name="entrenar_resnet_dann",
            ),
            # MDD
            node(
                entrenar_lenet_mdd,
                inputs=None,
                outputs="modelo_lenet_mdd",
                name="entrenar_lenet_mdd",
            ),
            node(
                entrenar_resnet_mdd,
                inputs=None,
                outputs="modelo_resnet_mdd",
                name="entrenar_resnet_mdd",
            ),
            # AFN
            node(
                entrenar_lenet_afn,
                inputs=None,
                outputs="modelo_lenet_afn",
                name="entrenar_lenet_afn",
            ),
            node(
                entrenar_resnet_afn,
                inputs=None,
                outputs="modelo_resnet_afn",
                name="entrenar_resnet_afn",
            ),
            # node(
            #     aplicar_modelo,
            #     inputs=["modelo_lenet_adda", "dataset_telegramas"],
            #     outputs="modelo_lenet_adda_predicciones",
            #     name="modelo_lenet_adda_aplicar_modelo",
            # ),
            # node(
            #     aplicar_modelo,
            #     inputs=["modelo_lenet_source_only", "dataset_telegramas"],
            #     outputs="modelo_lenet_source_only_predicciones",
            #     name="modelo_lenet_source_only_aplicar_modelo",
            # ),
            # node(
            #     aplicar_modelo,
            #     inputs=["modelo_lenet_target_only", "dataset_telegramas"],
            #     outputs="modelo_lenet_target_only_predicciones",
            #     name="modelo_lenet_target_only_aplicar_modelo",
            # ),
            # node(
            #     aplicar_modelo,
            #     inputs=["modelo_lenet_dann", "dataset_telegramas"],
            #     outputs="modelo_lenet_dann_predicciones",
            #     name="modelo_lenet_dann_aplicar_modelo",
            # ),
            # node(
            #     aplicar_modelo,
            #     inputs=["modelo_lenet_mdd", "dataset_telegramas"],
            #     outputs="modelo_lenet_mdd_predicciones",
            #     name="modelo_lenet_mdd_aplicar_modelo",
            # ),
            # node(
            #     aplicar_modelo,
            #     inputs=["modelo_lenet_afn", "dataset_telegramas"],
            #     outputs="modelo_lenet_afn_predicciones",
            #     name="modelo_lenet_afn_aplicar_modelo",
            # ),
            # node(
            #     aplicar_modelo,
            #     inputs=["modelo_lenet_afn", "dataset_telegramas"],
            #     outputs="modelo_lenet_afn_predicciones",
            #     name="modelo_lenet_afn_aplicar_modelo",
            # ),
        ]
    )
