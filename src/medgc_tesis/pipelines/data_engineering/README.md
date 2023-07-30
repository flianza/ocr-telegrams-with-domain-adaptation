# Pipeline data_engineering

Este pipeline se encarga de tomar todos los telegramas, crear el dataset TDS segmentando los 
dígitos y armando un dataset en formato DataFrame de pandas.

## Overview

El pipeline consta de los siguientes pasos:

- Extraccion de dígitos: por cada uno de los telgramas, realiza las siguientes tareas.
    - Endereza el telegrama mediante la función `deskew` definida en `image_utils/angle.py`.
    - Escala el telegrama a 1700px x 2800px usando openCV.
    - Recorta la grilla de votos mediante la función `crop_largest_rectagle` definida en `image_utils/crop.py`.
    - Busca las líneas de la grilla mediante la función `buscar_lineas_rectas` definida en `image_utils/lines.py`.
    - Extrae los dígitos de los votos mediante la función `extraer_votos` definida en `image_utils/segmentation.py`.
- Armar dataset: arma el dataset de pandas donde cada registro representa un voto de senadores o 
diputados con las imágenes de los dígitos. 
    - Limpia el dataset de aquellos telegramas que hayan sido mal preprocesados o mal cargados. 
    - Limpia aquellas imágenes que no sean dígitos mediante filtros de tamaño, proporción de pixeles 
    blancos, etc.
- Guardar TDS: guarda las imágenes de los dígitos extraídos en `data/05_model_input/TDS`.

## Pipeline inputs

- Los telegramas descargados desde la página oficial en `data/01_raw/telegramas`.

## Pipeline outputs

- Dataframe con el dataset.
- Conjunto de imagenes TDS.
