import os
from typing import Any, Tuple

from tllib.utils.data import ForeverDataIterator
from tllib.vision.datasets import MNIST, ImageList


class ForeverDataIteratorCustom(ForeverDataIterator):
    def __iter__(self):
        return self


class MNISTCustom(ImageList):
    image_list = {
        "train": "image_list/mnist_train.txt",
        "test": "image_list/mnist_test.txt",
        "val": "image_list/mnist_test.txt",
    }
    CLASSES = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(self, root, mode="L", split="train", **kwargs):
        assert split in ["train", "test", "val"]
        root = root.replace("\\", "/")
        data_list_file = f"{root}/{self.image_list[split]}"

        assert mode in ["L", "RGB"]
        self.mode = mode
        super().__init__(root, MNISTCustom.CLASSES, data_list_file=data_list_file, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        path, target = self.samples[index]
        img = self.loader(path).convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    @classmethod
    def get_classes(self):
        return MNIST.CLASSES


class TDSCustom(ImageList):
    CLASSES = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(self, root, mode="L", split="", **kwargs):
        root = root.replace("\\", "/")
        data_list_file = f"{root}/image_list/tds_{split}.txt"

        assert mode in ["L", "RGB"]
        self.mode = mode

        super().__init__(root, TDSCustom.CLASSES, data_list_file=data_list_file, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        path, target = self.samples[index]
        img = self.loader(path).convert(self.mode)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)

        return img, target
