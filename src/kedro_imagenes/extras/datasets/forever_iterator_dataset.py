import ssl
from pathlib import PurePosixPath
from typing import Any, Dict

import torchvision.transforms as T
from common.utils.data import ForeverDataIterator
from common.vision.datasets import MNIST
from common.vision.transforms import ResizeImage
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path
from torch.utils.data import DataLoader

from kedro_imagenes.extras.datasets.tds import TDS

ssl._create_default_https_context = ssl._create_unverified_context


def get_data_transform(
    random_horizontal_flip=False,
    random_color_jitter=False,
    resize_size=28,
    norm_mean=0.5,
    norm_std=0.5,
):
    transforms = [ResizeImage(resize_size)]

    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))

    transforms.extend([T.ToTensor(), T.Normalize(mean=norm_mean, std=norm_std)])

    return T.Compose(transforms)


class ForeverItaratorDataSet(AbstractDataSet):
    def __init__(self, path: str):
        protocol, full_path = get_protocol_and_path(path)
        self._protocol = protocol
        self._filepath = PurePosixPath(full_path)

        self.load_path = get_filepath_str(self._filepath, self._protocol)
        self.transform = get_data_transform()

        self.dataset = None

    def _load(self) -> ForeverDataIterator:
        dataset_loader = DataLoader(self.dataset, batch_size=32, shuffle=True, drop_last=True)

        return ForeverDataIterator(dataset_loader)

    def _save(self, data: Any) -> None:
        raise NotImplementedError("This dataset is read-only.")

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, protocol=self._protocol)


class MNISTDataSet(ForeverItaratorDataSet):
    def __init__(self, path: str, split: str):
        super().__init__(path)
        self.dataset = MNIST(self.load_path, split=split, transform=self.transform)


class TDSDataSet(ForeverItaratorDataSet):
    def __init__(self, path: str):
        super().__init__(path)
        self.dataset = TDS(self.load_path, transform=self.transform)
