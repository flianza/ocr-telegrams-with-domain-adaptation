from kedro.pipeline import Pipeline, node

from medgc_tesis.pipelines.modeling.nodes import analizar_modelo, entrenar_dann


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                entrenar_dann,
                inputs=["params:dann", "digitos_mnist_train", "digitos_mnist_test", "digitos_tds"],
                outputs="modelo_dann",
                name="train_dann",
            ),
            node(
                analizar_modelo,
                inputs=["modelo_dann", "digitos_mnist_train", "digitos_tds"],
                outputs=["modelo_dann_features_mnist", "modelo_dann_features_tds"],
                name="analizar_modelo",
            ),
        ]
    )
