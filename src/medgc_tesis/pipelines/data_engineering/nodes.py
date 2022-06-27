import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from medgc_tesis.pipelines.data_engineering.image_utils import (
    angle,
    crop,
    lines,
    segmentation,
)

logger = logging.getLogger(__name__)


def extraer_digitos(telegramas: Dict[str, Callable]) -> Iterable[Dict[str, Any]]:
    """Extrae los digitos de los telegramas.

    Parameters
    ----------
    telegramas : Dict[str, Callable]
        Dataset de telegramas.

    Returns
    -------
    Iterable[Dict[str, Any]]
        Nombre de los telegramas y su contenido.
    """
    telegramas_segmentados = []
    for nombre, telegrama_loader in tqdm(telegramas.items()):
        telegrama = telegrama_loader()
        try:
            telegrama = angle.deskew(telegrama)
            telegrama = cv2.resize(telegrama, (1700, 2800))
            telegrama = crop.crop_largest_rectagle(telegrama)

            xs = lines.buscar_lineas_rectas(telegrama, eje=lines.Eje.X)
            ys = lines.buscar_lineas_rectas(telegrama, eje=lines.Eje.Y)

            assert len(xs) == 4, f"{nombre}: Debe haber 4 lineas horizontales y hay {len(xs)}"
            assert len(ys) == 15, f"{nombre}: Debe haber 15 lineas verticales y hay {len(ys)}"

            # Vamos a eliminar la primer row (la que tiene los titulos)
            # y las ultimas 4 (votos blanco, nulos, recurridos, identidad impugnada)
            ys = ys[1:11]

            votos = segmentation.extraer_votos(telegrama, cortes_horizontales=xs, cortes_verticales=ys)
            telegramas_segmentados.append({"nombre": nombre, "votos": votos})

        except Exception:
            logger.error(f"Telegrama {nombre}\n{traceback.format_exc()}")

    return telegramas_segmentados


def armar_dataset(telegramas_segmentados: Iterable[Dict[str, Any]], mesas_escrutadas: pd.DataFrame) -> pd.DataFrame:
    """Genera un dataset con los telegramas segmentados.

    Parameters
    ----------
    telegramas_segmentados : Iterable[Dict[str, Any]]
        Telegramas segmentados.

    Returns
    -------
    pd.DataFrame
        Dataset que contiene un registro por voto de cada telegrama.
    """
    id_telegrama = []
    tipo = []
    digitos = []
    partido = []
    partidos = {
        0: "UNITE POR LA LIBERTAD Y LA DIGNIDAD",
        1: "FRENTE AMPLIO PROGRESISTA",
        2: "FRENTE DE IZQUIERDA Y DE TRABAJADORES - UNIDAD",
        3: "JUNTOS POR EL CAMBIO",
        4: "PRIMERO SANTA FE",
        5: "SOMOS FUTURO",
        6: "PODEMOS",
        7: "SOBERANIA POPULAR",
        8: "FRENTE DE TODOS",
    }

    for telegrama in telegramas_segmentados:
        for idx, votos in enumerate(telegrama["votos"]):
            id_telegrama.append(telegrama["nombre"])
            tipo.append("SENADORES NACIONALES")
            partido.append(partidos.get(idx, None))
            digitos.append(votos["senadores"])

            id_telegrama.append(telegrama["nombre"])
            tipo.append("DIPUTADOS NACIONALES")
            partido.append(partidos.get(idx, None))
            digitos.append(votos["diputados"])
    df = pd.DataFrame(
        {
            "id_telegrama": id_telegrama,
            "partido": partido,
            "tipo": tipo,
            "digitos": digitos,
        }
    )

    df["mesa"] = df.id_telegrama.str[-6:]
    df_mesas_santafe = (
        mesas_escrutadas.query("(Distrito == 'Santa Fe') & (tipoVoto == 'positivo')")
        .rename(columns={"Mesa": "mesa", "Agrupacion": "partido", "Cargo": "tipo"})[
            ["mesa", "tipo", "partido", "votos"]
        ]
        .copy()
    )
    df = pd.merge(df, df_mesas_santafe, on=["mesa", "tipo", "partido"])

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
    df = df.rename(columns={"digitos_escalados": "digitos"})

    return df


def guardar_digitos_separados(dataset: pd.DataFrame) -> None:
    """Separa los digitos de los telegramas y los guarda separados.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset de los telegramas y sus digitos.
    """
    todos_digitos = []
    dataset["votos"] = dataset.votos.astype(str)
    for row in dataset.itertuples():
        if len(row.votos) <= len(row.digitos):
            k = len(row.votos)
            for digito, numero in zip(row.digitos[-k:], row.votos[-k:]):
                todos_digitos.append((np.stack(digito, axis=0), numero))

    base_path = "./data/05_model_input/TDS"
    Path(f"{base_path}/image_list").mkdir(parents=True, exist_ok=True)
    Path(f"{base_path}/tds_train_image").mkdir(parents=True, exist_ok=True)
    Path(f"{base_path}/tds_val_image").mkdir(parents=True, exist_ok=True)
    Path(f"{base_path}/tds_test_image").mkdir(parents=True, exist_ok=True)

    train_digitos, test_digitos = train_test_split(todos_digitos, test_size=0.3, random_state=42)
    val_digitos, test_digitos = train_test_split(test_digitos, test_size=0.5, random_state=42)

    guardar_split_digitos(train_digitos, base_path, split="train")
    guardar_split_digitos(val_digitos, base_path, split="val", offset=len(train_digitos))
    guardar_split_digitos(
        test_digitos,
        base_path,
        split="test",
        offset=len(train_digitos) + len(val_digitos),
    )


def guardar_split_digitos(todos_digitos, base_path, split, offset=0):
    for idx, (digito, numero) in enumerate(tqdm(todos_digitos)):
        path_digito = f"tds_{split}_image/{idx + offset}.png"
        cv2.imwrite(f"{base_path}/{path_digito}", digito)

        with open(f"{base_path}/image_list/tds_{split}.txt", "a") as file:
            file.write(f"{path_digito} {numero}\n")
