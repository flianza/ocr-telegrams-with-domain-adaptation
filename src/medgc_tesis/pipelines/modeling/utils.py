import os
from typing import Any, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import umap
from tllib.utils.data import ForeverDataIterator
from tllib.vision.datasets import MNIST, ImageList
from tllib.vision.transforms import ResizeImage
from torch.utils.data import DataLoader


def get_dataloader(dataset, split, transform):
    if dataset == "MNIST":
        dataset = MNISTCustom(".\\data\\05_model_input\\MNIST", split=split, transform=transform)
    else:
        dataset = TDSCustom(".\\data\\05_model_input\\TDS", split=split, transform=transform)
    dataset_loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    return ForeverDataIteratorCustom(dataset_loader)


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
        data_list_file = os.path.join(root, self.image_list[split])

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
        data_list_file = os.path.join(root, f"image_list/tds_{split}.txt")

        assert mode in ["L", "RGB"]
        self.mode = mode

        super().__init__(root, TDSCustom.CLASSES, data_list_file=data_list_file, **kwargs)

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


class Backbone:
    def model(self):
        ...

    def data_transform(self):
        ...

    def pool_layer(self):
        ...


class LeNet(nn.Sequential):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.num_classes = num_classes
        self.out_features = 500

    def copy_head(self):
        return nn.Linear(500, self.num_classes)


class LeNetBackbone(Backbone):
    def model(self):
        return LeNet()

    def data_transform(self):
        return T.Compose([ResizeImage(28), T.ToTensor(), T.Normalize(mean=0.5, std=0.5)])

    def pool_layer(self):
        return nn.Identity()


def analyze_latent_space(source_latent_space: torch.Tensor, target_latent_space: torch.Tensor):
    import logging

    logger = logging.getLogger(__name__)

    logger.info("Calculando A-distance.")
    a_distance = calcular_a_distance(source_latent_space, target_latent_space)
    logger.info(f"A-distance: {a_distance:.4f}")

    logger.info("Aplicando UMAP.")
    df_features = aplicar_umap(source_latent_space, target_latent_space)
    logger.info("UMAP aplicado.")

    return a_distance, df_features


def aplicar_umap(source_latent_space, target_latent_space):
    mapper = umap.UMAP(random_state=33)
    source_feature = mapper.fit_transform(source_latent_space)
    target_feature = mapper.transform(target_latent_space)

    df_mnist = pd.DataFrame({"0": source_feature[:, 0], "1": source_feature[:, 1]})
    df_mnist["label"] = "MNIST"
    df_tds = pd.DataFrame({"0": target_feature[:, 0], "1": target_feature[:, 1]})
    df_tds["label"] = "TDS"
    df_features = pd.concat([df_mnist, df_tds], ignore_index=True)
    return df_features


def calcular_a_distance(source_latent_space, target_latent_space):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    df_source = pd.DataFrame(source_latent_space.numpy())
    df_source["y"] = 0
    df_target = pd.DataFrame(target_latent_space.numpy())
    df_target["y"] = 1
    df = pd.concat([df_source, df_target], ignore_index=True)

    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

    lr = LogisticRegression()
    lr.fit(df_train.drop(columns="y"), df_train.y)

    acc = accuracy_score(df_test.y, lr.predict(df_test.drop(columns="y")))
    error = 1 - acc
    a_distance = 2 * (1 - 2 * error)

    return a_distance
