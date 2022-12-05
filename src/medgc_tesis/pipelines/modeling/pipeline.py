from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from medgc_tesis.pipelines.modeling import nodes
from medgc_tesis.pipelines.modeling.nodes import aplicar_modelo


def create_steps(model_name, da_technique):
    train_func_name = f"entrenar_{model_name}_{da_technique}"
    train_func = getattr(nodes, train_func_name)

    return [
        node(
            train_func,
            inputs=None,
            outputs=f"modelo_{model_name}_{da_technique}",
            name=f"entrenar_{model_name}_{da_technique}",
        ),
        node(
            aplicar_modelo,
            inputs=[f"modelo_{model_name}_{da_technique}", "dataset_telegramas"],
            outputs=f"modelo_{model_name}_{da_technique}_predicciones",
            name=f"modelo_{model_name}_{da_technique}_aplicar_modelo",
        ),
    ]


def create_pipeline(**kwargs) -> Pipeline:

    pipes = []
    for modelo in ["resnet", "lenet"]:
        for da in ["bsp"]:
            pipes += create_steps(modelo, da)

    return pipeline(pipes)
