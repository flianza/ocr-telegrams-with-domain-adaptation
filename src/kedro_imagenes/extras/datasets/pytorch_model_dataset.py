from pathlib import PurePosixPath
from typing import Any, Dict

import torch
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path


class PytorchModelDataSet(AbstractDataSet):
    def __init__(self, filepath: str):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)

    def _load(self) -> Any:
        load_path = get_filepath_str(self._filepath, self._protocol)
        model = torch.load(load_path)
        return model

    def _save(self, model: Any) -> None:
        save_path = get_filepath_str(self._filepath, self._protocol)
        torch.save(model, save_path)

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, protocol=self._protocol)
