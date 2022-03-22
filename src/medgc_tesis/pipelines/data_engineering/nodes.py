import logging
import traceback
from typing import Any, Callable, Dict, Iterable

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .image_utils import angle, crop, lines, segmentation

logger = logging.getLogger(__name__)


def extraer_digitos(telegramas: Dict[str, Callable]) -> Iterable:
    telegramas_segmentados = []
    for nombre, telegrama_loader in tqdm(telegramas.items()):
        telegrama = telegrama_loader()
        try:
            telegrama = angle.deskew(telegrama)
            telegrama = crop.crop_largest_rectagle(telegrama)

            xs = lines.buscar_lineas_rectas(telegrama, eje=lines.Eje.X)
            ys = lines.buscar_lineas_rectas(telegrama, eje=lines.Eje.Y)

            assert len(xs) == 4, f"{nombre}: Debe haber 4 lineas horizontales y hay {len(xs)}"
            assert len(ys) <= 15, f"{nombre}: Debe haber como maximo 15 lineas verticales y hay {len(ys)}"

            # Vamos a eliminar la primer row (la que tiene los titulos)
            # y las ultimas 4 (votos blanco, nulos, recurridos, identidad impugnada)
            ys = ys[1:11]

            votos = segmentation.extraer_votos(telegrama, cortes_horizontales=xs, cortes_verticales=ys)
            telegramas_segmentados.append({"nombre": nombre, "votos": votos})

        except Exception:
            logger.error(f"Telegrama {nombre}\n{traceback.format_exc()}")

    return telegramas_segmentados


def armar_dataset(telegramas_segmentados: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    id_telegrama = []
    tipo = []
    digitos = []
    partido = []
    partidos = {
        0: "unite",
        1: "frente amplio progresista",
        2: "fit",
        3: "juntos",
        4: "primero santa fe",
        5: "somos futuro",
        6: "podemos",
        7: "soberania popular",
        8: "frente de todos",
    }

    for telegrama in telegramas_segmentados:
        for idx, votos in enumerate(telegrama["votos"]):
            id_telegrama.append(telegrama["nombre"])
            tipo.append("senadores")
            partido.append(partidos.get(idx, None))
            digitos.append(votos["senadores"])

            id_telegrama.append(telegrama["nombre"])
            tipo.append("diputados")
            partido.append(partidos.get(idx, None))
            digitos.append(votos["diputados"])
    df = pd.DataFrame({"id_telegrama": id_telegrama, "partido": partido, "tipo": tipo, "digitos": digitos})

    def proporcion_pixeles_blancos(digito: np.ndarray) -> float:
        return np.sum(digito) / (digito.shape[0] ** 2 * 255)

    def calcular_indicadores_digitos(df: pd.DataFrame) -> pd.DataFrame:
        df["cant_digitos"] = df.digitos.apply(lambda digitos: len(digitos))
        df["min_size_digitos"] = df.digitos.apply(
            lambda digitos: np.min([digito.shape[0] for digito in digitos] + [np.inf])
        )
        df["max_size_digitos"] = df.digitos.apply(
            lambda digitos: np.max([digito.shape[0] for digito in digitos] + [-np.inf])
        )
        df["min_prop_blanco_digitos"] = df.digitos.apply(
            lambda digitos: np.min([proporcion_pixeles_blancos(digito) for digito in digitos] + [np.inf])
        )
        df["max_prop_blanco_digitos"] = df.digitos.apply(
            lambda digitos: np.max([proporcion_pixeles_blancos(digito) for digito in digitos] + [-np.inf])
        )

        return df.replace([np.inf, -np.inf], np.nan)

    df = calcular_indicadores_digitos(df)

    # Segun el EDA, tenemos que filtrar por proporcion de pixeles blancos
    # y por el size de los digitos
    UMBRAL_PROP_BLANCO_MAXIMO = 0.95
    df["digitos"] = df.digitos.apply(
        lambda digitos: [digito for digito in digitos if proporcion_pixeles_blancos(digito) < UMBRAL_PROP_BLANCO_MAXIMO]
    )
    df = calcular_indicadores_digitos(df)

    UMBRAL_PROP_BLANCO_MINIMO = 0.5
    df = df.query(f"min_prop_blanco_digitos >= {UMBRAL_PROP_BLANCO_MINIMO}").copy()

    UMBRAL_SIZE_MAXIMO = np.mean(df.max_size_digitos) + 4 * np.std(df.max_size_digitos)
    df = df.query(f"max_size_digitos <= {UMBRAL_SIZE_MAXIMO}").copy()

    # Tenemos que filtrar aquellos telegramas que tengan mas de 3 digitos en alguno de sus votos
    df_cant_sospechoso = df.query("cant_digitos > 3").copy()
    df = df[~df.id_telegrama.isin(df_cant_sospechoso.id_telegrama.unique())].copy()

    # Tenemos que filtrar aquellos telegramas que no hayan sido leidos enteros
    id_telegramas_completos = (
        df.groupby(["id_telegrama"]).partido.count().reset_index().query("partido == 18").id_telegrama
    )
    df = df[df.id_telegrama.isin(id_telegramas_completos.values)]

    # Finalmente los escalamos a (28, 28) segun MNIST
    def escalar_digitos(digitos):
        digitos_ = []
        for digito in digitos:
            digito_escalado = cv2.resize(digito, (28, 28), interpolation=cv2.INTER_AREA)
            digito_escalado = 255 - digito_escalado
            digitos_.append(digito_escalado.tolist())
        return digitos_

    df["digitos_escalados"] = df.digitos.apply(escalar_digitos)
    df = df.drop(["digitos"], axis=1)

    return df
