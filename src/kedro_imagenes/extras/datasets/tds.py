import os
from typing import Any, Tuple

from common.vision.datasets.imagelist import ImageList


class TDS(ImageList):
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
        data_list_file = os.path.join(root, f"image_list/tds_{split}.txt")

        assert mode in ["L", "RGB"]
        self.mode = mode

        super(TDS, self).__init__(root, TDS.CLASSES, data_list_file=data_list_file, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index
        return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path).convert(self.mode)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)

        return img, target
