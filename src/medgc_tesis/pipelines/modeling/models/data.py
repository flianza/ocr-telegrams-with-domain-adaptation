import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader

from medgc_tesis.pipelines.modeling.data import (
    ForeverDataIteratorCustom,
    MNISTCustom,
    TDSCustom,
)


class DomainAdaptationDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1024, transform=None):
        super().__init__()
        self.data_dir = ".\\data\\05_model_input\\"
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage):
        self.mnist_train = MNISTCustom(f"{self.data_dir}MNIST", split="train", transform=self.transform)
        self.tds_train = TDSCustom(f"{self.data_dir}TDS", split="train", transform=self.transform)
        self.mnist_val = MNISTCustom(f"{self.data_dir}MNIST", split="val", transform=self.transform)
        self.tds_val = TDSCustom(f"{self.data_dir}TDS", split="val", transform=self.transform)
        self.mnist_test = MNISTCustom(f"{self.data_dir}MNIST", split="test", transform=self.transform)
        self.tds_test = TDSCustom(f"{self.data_dir}TDS", split="test", transform=self.transform)

    def train_dataloader(self):
        return CombinedLoader(
            {
                "mnist": ForeverDataIteratorCustom(
                    DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
                ),
                "tds": ForeverDataIteratorCustom(
                    DataLoader(self.tds_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
                ),
            }
        )

    def val_dataloader(self):
        return CombinedLoader(
            {
                "mnist": ForeverDataIteratorCustom(
                    DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=True, drop_last=True)
                ),
                "tds": ForeverDataIteratorCustom(
                    DataLoader(self.tds_val, batch_size=self.batch_size, shuffle=True, drop_last=True)
                ),
            }
        )

    def test_dataloader(self):
        return CombinedLoader(
            {
                "mnist": ForeverDataIteratorCustom(
                    DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=True, drop_last=True)
                ),
                "tds": ForeverDataIteratorCustom(
                    DataLoader(self.tds_test, batch_size=self.batch_size, shuffle=True, drop_last=True)
                ),
            }
        )
