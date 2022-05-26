# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =========================================================================
# Adapted from https://github.com/pytorch/vision/blob/release/0.12/torchvision/models/detection/anchor_utils.py
# which has the following license...
# https://github.com/pytorch/vision/blob/main/LICENSE

# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

"""
This script is adapted from
https://github.com/pytorch/vision/blob/release/0.12/torchvision/models/detection/anchor_utils.py
"""

from typing import List, Sequence, Union

import torch
from torch import Tensor, nn

from monai.utils import ensure_tuple
from monai.utils.module import look_up_option


class AnchorGenerator(nn.Module):
    """
    This module is modified from torchvision to support both 2D and 3D images.

    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements.
    For 2D images, anchor width and height w:h = 1:aspect_ratios[i,j]
    For 3D images, anchor width, height, and depth w:h:d = 1:aspect_ratios[i,j,0]:aspect_ratios[i,j,1]

    AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Args:
        sizes: base size of each anchor.
            len(sizes) is the number of feature maps, i.e., the number of output levels for
            the feature pyramid network (FPN).
            Each elment of ``sizes`` is a Sequence which represents several anchor sizes for each feature map.
        aspect_ratios: the aspect ratios of anchors. ``len(aspect_ratios) = len(sizes)``.
            For 2D images, each element of ``aspect_ratios[i]`` is a Sequence of float.
            For 3D images, each element of ``aspect_ratios[i]`` is a Sequence of 2 value Sequence.
        indexing: choose from {'xy', 'ij'}, optional
            Cartesian ('xy') or matrix ('ij', default) indexing of output.
            Cartesian ('xy') indexing swaps axis 0 and 1, which is the setting inside torchvision.
            matrix ('ij', default) indexing keeps the original axis not changed.
            See also indexing in https://pytorch.org/docs/stable/generated/torch.meshgrid.html

    Reference:.
        https://github.com/pytorch/vision/blob/release/0.12/torchvision/models/detection/anchor_utils.py

    Example:
        .. code-block:: python

            # 2D example inputs for a 2-level feature maps
            sizes = ((10,12,14,16), (20,24,28,32))
            base_aspect_ratios = (1., 0.5,  2.)
            aspect_ratios = (base_aspect_ratios, base_aspect_ratios)
            AnchorGenerator(sizes, aspect_ratios)

            # 3D example inputs for a 2-level feature maps
            sizes = ((10,12,14,16), (20,24,28,32))
            base_aspect_ratios = ((1., 1.), (1., 0.5), (0.5, 1.), (2., 2.))
            aspect_ratios = (base_aspect_ratios, base_aspect_ratios)
            AnchorGenerator(sizes, aspect_ratios)
    """

    __annotations__ = {"cell_anchors": List[torch.Tensor]}

    def __init__(
        self,
        sizes: Sequence[Sequence[int]] = ((20, 30, 40),),
        aspect_ratios: Sequence = (((0.5, 1), (1, 0.5)),),
        indexing: str = "ij",
    ) -> None:
        super().__init__()

        if not isinstance(sizes[0], Sequence):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], Sequence):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        if len(sizes) != len(aspect_ratios):
            raise ValueError(
                "len(sizes) and len(aspect_ratios) should be equal. \
                It represents the number of feature maps."
            )

        spatial_dims = len(ensure_tuple(aspect_ratios[0][0])) + 1
        spatial_dims = look_up_option(spatial_dims, [2, 3])
        self.spatial_dims = spatial_dims

        self.indexing = look_up_option(indexing, ["ij", "xy"])

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [
            self.generate_anchors(size, aspect_ratio) for size, aspect_ratio in zip(sizes, aspect_ratios)
        ]

    # This comment comes from torchvision.
    # TODO: https://github.com/pytorch/pytorch/issues/26792
    # For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
    # (scales, aspect_ratios) are usually an element of zip(self.scales, self.aspect_ratios)
    # This method assumes aspect ratio = height / width for an anchor.
    def generate_anchors(
        self,
        scales: Sequence,
        aspect_ratios: Sequence,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, None] = None,
    ) -> torch.Tensor:
        """
        Compute cell anchor shapes at multiple sizes and aspect ratios for the current feature map.

        Args:
            scales: a sequence which represents several anchor sizes for the current feature map.
            aspect_ratios: a sequence which represents several aspect_ratios for the current feature map.
                For 2D images, it is a Sequence of float aspect_ratios[j],
                anchor width and height w:h = 1:aspect_ratios[j].
                For 3D images, it is a Sequence of 2 value Sequence aspect_ratios[j,0] and aspect_ratios[j,1],
                anchor width, height, and depth w:h:d = 1:aspect_ratios[j,0]:aspect_ratios[j,1]
            dtype: target data type of the output Tensor.
            device: target device to put the output Tensor data.

            Returns:
                For each s in scales, returns [s, s*aspect_ratios[j]] for 2D images,
                and [s, s*aspect_ratios[j,0],s*aspect_ratios[j,1]] for 3D images.
        """
        if device is None:
            device = torch.device("cpu")
        scales_t = torch.as_tensor(scales, dtype=dtype, device=device)  # sized (N,)
        aspect_ratios_t = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)  # sized (M,) or (M,2)
        if (self.spatial_dims >= 3) and (len(aspect_ratios_t.shape) != 2):
            ValueError(
                f"In {self.spatial_dims}-D image, aspect_ratios for each level should be \
                {len(aspect_ratios_t.shape)-1}-D. But got aspect_ratios with shape {aspect_ratios_t.shape}."
            )

        if (self.spatial_dims >= 3) and (aspect_ratios_t.shape[1] != self.spatial_dims - 1):
            ValueError(
                f"In {self.spatial_dims}-D image, aspect_ratios for each level should has \
                shape (_,{self.spatial_dims-1}). But got aspect_ratios with shape {aspect_ratios_t.shape}."
            )

        # if 2d, w:h = 1:aspect_ratios
        if self.spatial_dims == 2:
            area_scale = torch.sqrt(aspect_ratios_t)
            w_ratios = 1 / area_scale
            h_ratios = area_scale
        # if 3d, w:h:d = 1:aspect_ratios[:,0]:aspect_ratios[:,1]
        elif self.spatial_dims == 3:
            area_scale = torch.pow(aspect_ratios_t[:, 0] * aspect_ratios_t[:, 1], 1 / 3.0)
            w_ratios = 1 / area_scale
            h_ratios = aspect_ratios_t[:, 0] / area_scale
            d_ratios = aspect_ratios_t[:, 1] / area_scale

        ws = (w_ratios[:, None] * scales_t[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales_t[None, :]).view(-1)
        if self.spatial_dims == 2:
            base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2.0
        elif self.spatial_dims == 3:
            ds = (d_ratios[:, None] * scales_t[None, :]).view(-1)
            base_anchors = torch.stack([-ws, -hs, -ds, ws, hs, ds], dim=1) / 2.0

        return base_anchors.round()

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        """
        Convert each element in self.cell_anchors to ``dtype`` and send to ``device``.
        """
        self.cell_anchors = [cell_anchor.to(dtype=dtype, device=device) for cell_anchor in self.cell_anchors]

    def num_anchors_per_location(self):
        """
        Return number of anchor shapes for each feature map.
        """
        return [c.shape[0] for c in self.cell_anchors]

    def grid_anchors(
        self, grid_sizes: Sequence[Sequence[int]], strides: Sequence[Sequence[Tensor]]
    ) -> Sequence[Tensor]:
        """
        Every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:spatial_dims)
        corresponds to a feature map.
        It outputs g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.

        Args:
            grid_sizes: spatial size of the feature maps
            strides: strides of the feature maps regarding to the original image

        Example:
            .. code-block:: python

                grid_sizes = [[100,100],[50,50]]
                strides = [[torch.tensor(2),torch.tensor(2)], [torch.tensor(4),torch.tensor(4)]]
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        if not (len(grid_sizes) == len(strides) == len(cell_anchors)):
            raise ValueError(
                "Anchors should be Tuple[Tuple[int]] because each feature "
                "map could potentially have different sizes and aspect ratios. "
                "There needs to be a match between the number of "
                "feature maps passed and the number of sizes / aspect ratios specified."
            )

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            # for each feature map
            device = base_anchors.device

            # compute anchor centers regarding to the image.
            # shifts_centers is [x_center, y_center] or [x_center, y_center, z_center]
            shifts_centers = []
            for axis in range(self.spatial_dims):
                shifts_centers.append(torch.arange(0, size[axis], dtype=torch.int32, device=device) * stride[axis])

            shifts_centers = list(torch.meshgrid(*tuple(shifts_centers), indexing="ij"))  # indexing="ij"
            for axis in range(self.spatial_dims):
                # each element of shifts_centers is sized (HW,) or (HWD,)
                shifts_centers[axis] = shifts_centers[axis].reshape(-1)

            # Expand to [x_center, y_center, x_center, y_center],
            # or [x_center, y_center, z_center, x_center, y_center, z_center]
            if self.indexing == "xy":
                # Cartesian ('xy') indexing swaps axis 0 and 1.
                shifts_centers[1], shifts_centers[0] = shifts_centers[0], shifts_centers[1]
            shifts = torch.stack(shifts_centers * 2, dim=1)  # sized (HW,4) or (HWD,6)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, self.spatial_dims * 2) + base_anchors.view(1, -1, self.spatial_dims * 2)).reshape(
                    -1, self.spatial_dims * 2
                )  # each element sized (AHWD,4) or (AHWD,6)
            )

        return anchors

    def forward(self, images: Tensor, feature_maps: List[Tensor]) -> List[Tensor]:
        """
        Generate anchor boxes for each image.

        Args:
            images: sized (B, C, W, H) or (B, C, W, H, D)
            feature_maps: for FPN level i, feature_maps[i] is sizec (B, C_i, W_i, H_i) or (B, C_i, W_i, H_i, D_i).
                This input argument does not have to be the actual feature maps.
                Any list variable with the same (C_i, W_i, H_i) or (C_i, W_i, H_i, D_i) as feature maps works.

        Return:
            A list with length of B. Each element represents the anchors for this image.
            The B elements are identical.

        Example:
            .. code-block:: python

                images = torch.zeros((3,1,128,128,128))
                feature_maps = [torch.zeros((3,6,64,64,32)), torch.zeros((3,6,32,32,16))]
        """
        grid_sizes = [feature_map.shape[-self.spatial_dims :] for feature_map in feature_maps]
        image_size = images.shape[-self.spatial_dims :]
        batchsize = images.shape[0]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [
                torch.tensor(image_size[axis] // g[axis], dtype=torch.int64, device=device)
                for axis in range(self.spatial_dims)
            ]
            for g in grid_sizes
        ]

        # Code below come from torchvision.models.detection.anchor_utils.AnchorGenerator.forward()
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)

        anchors: List[List[torch.Tensor]] = []
        for _ in range(batchsize):
            anchors_in_image = list(anchors_over_all_feature_maps)
            anchors.append(anchors_in_image)
        return [torch.cat(anchors_per_image) for anchors_per_image in anchors]


class AnchorGeneratorWithAnchorShape(AnchorGenerator, nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes, inherited from :py:class:`~monai.apps.detection.networks.utils.anchor_utils.AnchorGenerator`

    The module support computing anchors at multiple base anchor shapes
    per feature map.

    ``feature_map_scales`` should have the same number of elements with the number of feature maps.

    base_anchor_shapes can have an arbitrary number of elements.
    For 2D images, each element represents anchor width and height [w,h].
    For 2D images, each element represents anchor width, height, and depth [w,h,d].

    AnchorGenerator will output a set of ``len(base_anchor_shapes)`` anchors
    per spatial location for feature map ``i``.

    Args:
        feature_map_scales: scale of anchors for each feature map, i.e., each output level of
            the feature pyramid network (FPN). ``len(feature_map_scales)`` is the number of feature maps.
            ``scale[i]*base_anchor_shapes`` represents the anchor shapes for feature map ``i``.
        base_anchor_shapes: a sequence which represents several anchor shapes for one feature map.
            For N-D images, it is a Sequence of N value Sequence.
        indexing: choose from {'xy', 'ij'}, optional
            Cartesian ('xy') or matrix ('ij', default) indexing of output.
            Cartesian ('xy') indexing swaps axis 0 and 1, which is the setting inside torchvision.
            matrix ('ij', default) indexing keeps the original axis not changed.
            See also indexing in https://pytorch.org/docs/stable/generated/torch.meshgrid.html

    Example:
        .. code-block:: python

            # 2D example inputs for a 2-level feature maps
            feature_map_scales = (1, 2)
            base_anchor_shapes = ((10, 10), (6, 12), (12, 6))
            AnchorGeneratorWithAnchorShape(feature_map_scales, base_anchor_shapes)

            # 3D example inputs for a 2-level feature maps
            feature_map_scales = (1, 2)
            base_anchor_shapes = ((10, 10, 10), (12, 12, 8), (10, 10, 6), (16, 16, 10))
            AnchorGeneratorWithAnchorShape(feature_map_scales, base_anchor_shapes)
    """

    __annotations__ = {"cell_anchors": List[torch.Tensor]}

    def __init__(
        self,
        feature_map_scales: Union[Sequence[int], Sequence[float]] = (1, 2, 4, 8),
        base_anchor_shapes: Union[Sequence[Sequence[int]], Sequence[Sequence[float]]] = (
            (32, 32, 32),
            (48, 20, 20),
            (20, 48, 20),
            (20, 20, 48),
        ),
        indexing: str = "ij",
    ) -> None:
        nn.Module.__init__(self)

        spatial_dims = len(base_anchor_shapes[0])
        spatial_dims = look_up_option(spatial_dims, [2, 3])
        self.spatial_dims = spatial_dims

        self.indexing = look_up_option(indexing, ["ij", "xy"])

        base_anchor_shapes_t = torch.Tensor(base_anchor_shapes)
        self.cell_anchors = [self.generate_anchors_using_shape(s * base_anchor_shapes_t) for s in feature_map_scales]

    @staticmethod
    def generate_anchors_using_shape(
        anchor_shapes: torch.Tensor, dtype: torch.dtype = torch.float32, device: Union[torch.device, None] = None
    ) -> torch.Tensor:
        """
        Compute cell anchor shapes at multiple sizes and aspect ratios for the current feature map.

        Args:
            anchor_shapes: [w, h] or [w, h, d], sized (N, spatial_dims),
                represents N anchor shapes for the current feature map.
            dtype: target data type of the output Tensor.
            device: target device to put the output Tensor data.

        Returns:
            For 2D images, returns [-w/2, -h/2, w/2, h/2];
            For 3D images, returns [-w/2, -h/2, -d/2, w/2, h/2, d/2]
        """
        if device is None:
            device = torch.device("cpu")
        half_anchor_shapes = anchor_shapes / 2.0
        base_anchors = torch.cat([-half_anchor_shapes, half_anchor_shapes], dim=1)
        return base_anchors.round().to(dtype).to(device)
