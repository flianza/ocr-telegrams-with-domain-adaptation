import logging
import traceback
from typing import Callable, Dict, Iterable

from tqdm import tqdm

from .image_utils import angle, crop, lines, segmentation

logger = logging.getLogger(__name__)


def extraer_digitos(telegramas: Dict[str, Callable]) -> Iterable:
    dataset = []
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

        except Exception:
            logger.error(f"Telegrama {nombre}\n{traceback.format_exc()}")

        dataset.append({"nombre": nombre, "votos": votos})

    return dataset
