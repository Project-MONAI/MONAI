from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from monai.networks.nets import RegUNet


class GlobalNet(RegUNet):
    """
    Build GlobalNet for image registration.

    Reference:
        Hu, Yipeng, et al.
        "Label-driven weakly-supervised learning
        for multimodal deformable image registration,"
        https://arxiv.org/abs/1711.01666
    """

    def __init__(
        self,
        image_size: Union[Tuple[int], List[int]],
        spatial_dims: int,
        in_channels: int,
        num_channel_initial: int,
        depth: int,
        out_kernel_initializer: Optional[str] = "kaiming_uniform",
        out_activation: Optional[str] = None,
        pooling: bool = True,
        concat_skip: bool = False,
        encode_kernel_sizes: Union[int, List[int]] = 3,
    ):
        for size in image_size:
            if size % (2 ** depth) != 0:
                raise ValueError(
                    f"given extract_max_level {self.extract_max_level}, "
                    f"all input spatial dimension must be divisible by {2 ** self.extract_max_level}, "
                    f"got input of size {image_size}"
                )
        self.image_size = image_size
        self.decode_size = [int(size / (2 ** depth)) for size in image_size]
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            depth=depth,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            out_channels=spatial_dims,
            pooling=pooling,
            concat_skip=concat_skip,
            encode_kernel_sizes=encode_kernel_sizes,
        )

    def build_output_block(self):
        return AffineHead(
            spatial_dims=self.spatial_dims,
            image_size=self.image_size,
            decode_size=self.decode_size,
            in_channels=self.num_channels[-1],
        )


class AffineHead(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        image_size: Union[Tuple[int], List[int]],
        decode_size: Union[Tuple[int], List[int]],
        in_channels: int,
    ):
        super(AffineHead, self).__init__()
        self.spatial_dims = spatial_dims
        if spatial_dims == 2:
            in_features = in_channels * decode_size[0] * decode_size[1]
            out_features = 6
        elif spatial_dims == 3:
            in_features = in_channels * decode_size[0] * decode_size[1] * decode_size[2]
            out_features = 12
        else:
            raise ValueError(f"only support 2D/3D operation, got spatial_dims={spatial_dims}")

        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.grid = self.get_reference_grid(image_size)  # (spatial_dims, ...)

    @staticmethod
    def get_reference_grid(image_size: Union[Tuple[int], List[int]]) -> torch.Tensor:
        mesh_points = [torch.arange(0, dim) for dim in image_size]
        grid = torch.stack(torch.meshgrid(*mesh_points), dim=0)  # (spatial_dims, ...)
        return grid.to(dtype=torch.float)

    def affine_transform(self, theta: torch.Tensor):
        # (spatial_dims, ...) -> (spatial_dims + 1, ...)
        grid_padded = torch.cat([self.grid, torch.ones_like(self.grid[:1])])

        # grid_warped[b,p,...] = sum_over_q(grid_padded[q,...] * theta[b,p,q]
        if self.spatial_dims == 2:
            grid_warped = torch.einsum("qij,bpq->bpij", grid_padded, theta.reshape(-1, 2, 3))
        elif self.spatial_dims == 3:
            grid_warped = torch.einsum("qijk,bpq->bpijk", grid_padded, theta.reshape(-1, 3, 4))
        else:
            raise ValueError(f"do not support spatial_dims={self.spatial_dims}")
        return grid_warped

    def forward(self, x: List[torch.Tensor], image_size: List[int]) -> torch.Tensor:
        x = x[0]
        self.grid.to(x)
        theta = self.fc(x.reshape(x.shape[0], -1))
        out = self.affine_transform(theta) - self.grid
        return out
