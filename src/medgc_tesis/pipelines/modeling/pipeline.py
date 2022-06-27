from typing import Callable

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from medgc_tesis.pipelines.modeling.nodes import (
    aplicar_modelo,
    aplicar_umap,
    entrenar_adda,
    entrenar_afn,
    entrenar_dann,
    entrenar_vanilla,
    extraer_features,
    graficar_umap,
)


def create_subpipeline(type: str, train_func: Callable, datasets=None) -> Pipeline:
    if datasets is None:
        datasets = [
            "digitos_mnist_train",
            "digitos_tds_train",
            "digitos_tds_test",
            "digitos_tds_val",
        ]
    return pipeline(
        [
            node(
                train_func,
                inputs=[f"params:{type}"] + datasets,
                outputs=[f"modelo_{type}", f"modelo_{type}_history", f"modelo_{type}_metrics"],
                name=f"entrenar_{type}",
            ),
            node(
                extraer_features,
                inputs=[f"modelo_{type}", "digitos_mnist_test", "digitos_tds_test"],
                outputs=[f"modelo_{type}_features_mnist", f"modelo_{type}_features_tds"],
                name=f"modelo_{type}_extraer_features",
            ),
            node(
                aplicar_umap,
                inputs=[f"modelo_{type}_features_mnist", f"modelo_{type}_features_tds"],
                outputs=f"modelo_{type}_features_umap",
                name=f"modelo_{type}_aplicar_umap",
            ),
            node(
                graficar_umap,
                inputs=[f"modelo_{type}_features_umap"],
                outputs=f"modelo_{type}_plot_umap",
                name=f"modelo_{type}_graficar_umap",
            ),
            node(
                aplicar_modelo,
                inputs=[f"modelo_{type}", "dataset_telegramas"],
                outputs=f"modelo_{type}_predicciones",
                name=f"modelo_{type}_aplicar_modelo",
            ),
        ],
    )


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_dann = create_subpipeline("dann", entrenar_dann)
    pipeline_afn = create_subpipeline("afn", entrenar_afn)
    pipeline_adda = create_subpipeline("adda", entrenar_adda)

    pipeline_source_only = create_subpipeline(
        "source_only",
        entrenar_vanilla,
        datasets=["digitos_mnist_train", "digitos_tds_test", "digitos_tds_val"],
    )
    pipeline_target_only = create_subpipeline(
        "target_only",
        entrenar_vanilla,
        datasets=["digitos_tds_train", "digitos_tds_test", "digitos_tds_val"],
    )

    return pipeline_dann + pipeline_afn + pipeline_adda + pipeline_source_only + pipeline_target_only
