# MEDGC Tesis

Repositorio para la tesis de maestría "Clasificación de los dígitos escritos en los telegramas de 
las elecciones legislativas en Santa Fe mediante técnicas de adaptación de dominio".

## Overview

El proyecto se implemento en formato de pipelines mediante [kedro](https://kedro.org/).

La carpeta `docs` contiene el codigo latex de la tesis y del proyecto de tesis.

En la carpeta `src/medgc_tesis` se encuentran los pipelines del proyecto:

- `data_engineering`: se encarga de pre-procesar los datos y armar los inputs para entrenar 
los modelos.
- `modeling`: se encarga de entrenar distintos modelos con los telegramas pre-procesados.

En la carpeta `notebooks` se encuentran distintos EDAs.

En la carpeta `data` se almacenarán los datos que se vayan utilizando en cada pipeline.

En `conf/base/catalog.yml` se encuentra el catálogo de kedro donde se configuran los dataset
y su manera de almacenamiento.