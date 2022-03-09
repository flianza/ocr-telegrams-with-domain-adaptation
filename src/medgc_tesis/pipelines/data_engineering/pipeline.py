from kedro.pipeline import Pipeline, node

from medgc_tesis.pipelines.data_engineering.nodes import extraer_digitos


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                extraer_digitos,
                inputs=["telegramas"],
                outputs="votos_segmentados",
                name="extraer_digitos",
            )
        ]
    )
