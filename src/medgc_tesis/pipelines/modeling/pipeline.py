from typing import Callable

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from medgc_tesis.pipelines.modeling.nodes import (
    aplicar_modelo,
    entrenar_adda,
    entrenar_afn,
    entrenar_dann,
    entrenar_mdd,
    entrenar_source_only,
    entrenar_target_only,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                entrenar_adda,
                inputs=None,
                outputs="modelo_adda",
                name="entrenar_adda",
            ),
            node(
                entrenar_source_only,
                inputs=None,
                outputs="modelo_source_only",
                name="entrenar_source_only",
            ),
            node(
                entrenar_target_only,
                inputs=None,
                outputs="modelo_target_only",
                name="entrenar_target_only",
            ),
            node(
                entrenar_dann,
                inputs=None,
                outputs="modelo_dann",
                name="entrenar_dann",
            ),
            node(
                entrenar_mdd,
                inputs=None,
                outputs="modelo_mdd",
                name="entrenar_mdd",
            ),
            node(
                entrenar_afn,
                inputs=None,
                outputs="modelo_afn",
                name="entrenar_afn",
            ),
            # node(
            #     aplicar_modelo,
            #     inputs=["modelo_adda", "dataset_telegramas"],
            #     outputs="modelo_adda_predicciones",
            #     name="modelo_adda_aplicar_modelo",
            # ),
        ]
    )
