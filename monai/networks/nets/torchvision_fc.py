import torch
from torchvision import models


class TorchVisionFullyConvModel(torch.nn.Module):
    """
    Customize TorchVision models to replace fully connected layer by convolutional layer.

    Args:
        model_name: name of any torchvision with adaptive avg pooling and fully connected layer at the end.
            Default to "resnet18".
        n_classes: number of classes for the last classification layer. Default to 1.
        pretrained: whether to use the imagenet pretrained weights. Default to False.
    """

    def __init__(self, model_name: str = "resnet18", n_classes: int = 1, pretrained: bool = False):
        super().__init__()
        model = getattr(models, model_name)(pretrained=pretrained)

        # remove last fully connected (FC) layer and adaptive avg pooling
        self.features = torch.nn.Sequential(*list(model.children())[:-2])

        # add 7x7 avg pooling (in place of adaptive avg pooling)
        self.pool = torch.nn.AvgPool2d(7, stride=1)

        # add 1x1 conv (it behaves like a FC layer)
        self.fc = torch.nn.Conv2d(model.fc.in_features, n_classes, kernel_size=(1, 1))

    def forward(self, x):
        x = self.features(x)

        # apply 7x7 avg pooling
        x = self.pool(x)

        # apply last 1x1 conv layer that act like fc layer
        x = self.fc(x)

        # remove the color channel
        x = x.squeeze(1)
        return x
