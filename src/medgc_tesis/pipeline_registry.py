from typing import Dict

from kedro.pipeline import Pipeline

from medgc_tesis.pipelines import data_engineering, evaluation, modeling


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_engineering_pipeline = data_engineering.create_pipeline()
    modeling_pipeline = modeling.create_pipeline()
    evaluation_pipeline = evaluation.create_pipeline()

    return {
        "de": data_engineering_pipeline,
        "modeling": modeling_pipeline,
        "evaluation": evaluation_pipeline,
        "__default__": data_engineering_pipeline + modeling_pipeline + evaluation_pipeline,
    }
