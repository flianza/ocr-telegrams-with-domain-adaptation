# Pipeline modeling

Este pipeline se encarga de entrenar los modelos utilizando distintas técnicas de adaptación de dominio.

## Overview

Se entrenan redes Lenet5 y Resnet18, definidas en `backbone.py`.

Las técnicas de DA implementadas son:
- ADDA
- AFN
- DANN
- MDD
- BSP
Además, se implementa source only y target only a modo de benchmark.

Por cada modelo y técnica de adaptación se ejecutan las siguientes funciones definidas en `nodes.py`:
- `entrenar_{MODELO}_{DA}`: se encarga de buscar los mejores hiperparametros con Optuna y entrena el mejor modelo, 
guardándolo en `data/06_models/{DA}/{MODEL}`.
- `aplicar_modelo`: se encarga de tomar el modelo entrenado y lo aplica a todos los telegramas, calculando sus métricas
y dejándolas guardadas en `data/06_models/{DA}/{MODEL}`.


## Pipeline inputs

- El dataset armado en el pipeline de `data_engineering`.

## Pipeline outputs

- Modelos entrenados y métricas de cada uno.
