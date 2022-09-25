import torch.nn as nn
import torchvision.transforms as T
from tllib.vision.models import resnet18
from tllib.vision.transforms import ResizeImage


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


class Backbone:
    name: str

    def model(self):
        ...

    def data_transform(self):
        ...

    def pool_layer(self):
        ...


class LeNetBackbone(Backbone):
    name = "lenet"

    def model(self):
        return LeNet()

    def data_transform(self):
        return T.Compose([ResizeImage(28), T.ToTensor(), T.Normalize(mean=0.5, std=0.5)])

    def pool_layer(self):
        return nn.Identity()


class ResNet18Backbone(Backbone):
    name = "resnet"

    def model(self):
        return resnet18()

    def data_transform(self):
        return T.Compose(
            [
                T.Resize(32),
                T.ToTensor(),
                T.Lambda(lambda x: x.repeat(3, 1, 1)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
