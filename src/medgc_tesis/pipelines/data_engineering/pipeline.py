from kedro.pipeline import Pipeline, node

from medgc_tesis.pipelines.data_engineering.nodes import armar_dataset, extraer_digitos


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
        ]
    )
