from kedro.pipeline import Pipeline, node

from medgc_tesis.pipelines.data_engineering.nodes import (
    armar_dataset,
    extraer_digitos,
    guardar_digitos_separados,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                extraer_digitos,
                inputs=["telegramas"],
                outputs="telegramas_segmentados",
                name="extraer_digitos",
            ),
            node(
                armar_dataset,
                inputs=["telegramas_segmentados"],
                outputs="dataset_telegramas",
                name="armar_dataset",
            ),
            node(
                guardar_digitos_separados,
                inputs=["dataset_telegramas"],
                outputs=None,
                name="guardar_digitos_separados",
            ),
        ]
    )
