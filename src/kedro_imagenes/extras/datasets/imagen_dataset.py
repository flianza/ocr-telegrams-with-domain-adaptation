from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import cv2
import numpy as np
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path


class ImagenDataSet(AbstractDataSet):
    def __init__(self, filepath: str):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> np.array:
        load_path = get_filepath_str(self._filepath, self._protocol)
        imagen = cv2.imread(load_path)
        return imagen

    def _save(self, data: np.array) -> None:
        save_path = get_filepath_str(self._filepath, self._protocol)
        cv2.imwrite(save_path, data)

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, protocol=self._protocol)