from kedro.pipeline import Pipeline, node

from medgc_tesis.pipelines.modeling.nodes import (
    aplicar_modelo,
    aplicar_umap,
    entrenar_afn,
    entrenar_dann,
    extraer_features,
    graficar_umap,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                entrenar_dann,
                inputs=[
                    "params:dann",
                    "digitos_mnist_train",
                    "digitos_tds_train",
                    "digitos_tds_test",
                    "digitos_tds_val",
                ],
                outputs=["modelo_dann", "modelo_dann_history", "modelo_dann_metrics"],
                name="modelo_dann_train_dann",
            ),
            node(
                extraer_features,
                inputs=["modelo_dann", "digitos_mnist_train", "digitos_tds_train"],
                outputs=["modelo_dann_features_mnist", "modelo_dann_features_tds"],
                name="modelo_dann_extraer_features",
            ),
            node(
                aplicar_umap,
                inputs=["modelo_dann_features_mnist", "modelo_dann_features_tds"],
                outputs="modelo_dann_features_umap",
                name="modelo_dann_aplicar_umap",
            ),
            node(
                graficar_umap,
                inputs=["modelo_dann_features_umap"],
                outputs="modelo_dann_plot_umap",
                name="modelo_dann_graficar_umap",
            ),
            node(
                aplicar_modelo,
                inputs=["modelo_dann", "dataset_telegramas"],
                outputs="modelo_dann_predicciones",
                name="modelo_dann_aplicar_modelo",
            ),
            node(
                entrenar_afn,
                inputs=[
                    "params:afn",
                    "digitos_mnist_train",
                    "digitos_tds_train",
                    "digitos_tds_test",
                    "digitos_tds_val",
                ],
                outputs=["modelo_afn", "modelo_afn_history", "modelo_afn_metrics"],
                name="train_afn",
            ),
            node(
                extraer_features,
                inputs=["modelo_afn", "digitos_mnist_train", "digitos_tds_train"],
                outputs=["modelo_afn_features_mnist", "modelo_afn_features_tds"],
                name="modelo_afn_extraer_features",
            ),
            node(
                aplicar_umap,
                inputs=["modelo_afn_features_mnist", "modelo_afn_features_tds"],
                outputs="modelo_afn_features_umap",
                name="modelo_afn_aplicar_umap",
            ),
            node(
                graficar_umap,
                inputs=["modelo_afn_features_umap"],
                outputs="modelo_afn_plot_umap",
                name="modelo_afn_graficar_umap",
            ),
            node(
                aplicar_modelo,
                inputs=["modelo_afn", "dataset_telegramas"],
                outputs="modelo_afn_predicciones",
                name="modelo_afn_aplicar_modelo",
            ),
        ]
    )
