import monai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    def __init__(self, output_classes, device):
        super(BoundaryLoss, self).__init__()

        self.output_classes = output_classes

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
        ).to(device)

        self.conv2 = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
        ).to(device)

        weight = np.zeros(shape=(1, 1, 3, 3, 3), dtype=np.float32)
        weight[..., 0, :, :] = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32
        )
        weight[..., 1, :, :] = np.array(
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]], dtype=np.float32
        )
        weight[..., 2, :, :] = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32
        )

        with torch.no_grad():
            self.conv1.weight.copy_(
                torch.from_numpy(
                    1.0 / 27.0 * np.ones(shape=weight.shape, dtype=np.float32)
                )
            )
            self.conv2.weight.copy_(torch.from_numpy(weight))

        self.conv1.weight.requires_grad = False
        self.conv2.weight.requires_grad = False

    def compute_boundary(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        boundary_loss = 0

        for _i in range(1, self.output_classes):
            output_boundary = self.compute_boundary(output[:, _i : _i + 1, ...])
            target_boundary = self.compute_boundary(target[:, _i : _i + 1, ...])
            loss_value = torch.square(output_boundary - target_boundary)
            boundary_loss += loss_value

        boundary_loss = boundary_loss.mean()
        return boundary_loss

