from enum import Enum
from typing import Dict, Iterable

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


class Eje(Enum):
    X = 0
    Y = 1


RANGOS_CLUSTER = {
    Eje.X: range(4, 5),
    Eje.Y: range(10, 20),
}


def _cluster_projection(
    projection: np.ndarray, cluster_range: Iterable[int]
) -> np.ndarray:
    """
    Genera un agrupamiento optimo segun el silhouette_score de la proyeccion
    """
    best_cluster_model = None
    best_score = 0
    projection_ = projection.reshape(-1, 1)

    for n in cluster_range:
        cluster_model = AgglomerativeClustering(n_clusters=n)
        cluster_model.fit(projection_)
        score = silhouette_score(projection_, cluster_model.labels_)
        if score > best_score:
            best_score = score
            best_cluster_model = cluster_model

    return best_cluster_model.labels_


def _mean_by_cluster(items: np.ndarray, clusters: np.ndarray) -> Dict:
    """
    Calcula el valor medio de cada cluster
    """
    means = {}

    for cluster in np.unique(clusters):
        idxs = np.where(clusters == cluster)
        cluster_items = items[idxs]
        cluster_mean = np.mean(cluster_items)
        means[cluster] = min(cluster_items, key=lambda x: abs(x - cluster_mean))

    return means


def buscar_lineas_rectas(imagen: np.ndarray, eje: Eje) -> Iterable[int]:
    proyeccion_eje = np.sum(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY), eje.value)
    umbral_eje = np.mean(proyeccion_eje) - 2 * np.std(proyeccion_eje)
    pixeles_negros = np.where(proyeccion_eje < umbral_eje)[0]

    clusters = _cluster_projection(pixeles_negros, RANGOS_CLUSTER[eje])
    promedios_por_cluster = _mean_by_cluster(pixeles_negros, clusters)

    return sorted(promedios_por_cluster.values())
