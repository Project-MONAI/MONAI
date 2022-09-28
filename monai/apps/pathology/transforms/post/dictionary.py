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

from turtle import width
from typing import Hashable, Mapping, Optional, Sequence, Union

import torch

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.networks.nets import HoVerNet
from monai.transforms.transform import MapTransform
from monai.utils import optional_import

from .array import GenerateSuccinctContour, PostProcessHoVerNetOutput

find_contours, _ = optional_import("skimage.measure", name="find_contours")
moments, _ = optional_import("skimage.measure", name="moments")

__all__ = [
    "GenerateSuccinctContourDict",
    "GenerateSuccinctContourD",
    "GenerateSuccinctContourd",
    "PostProcessHoVerNetOutputDict",
    "PostProcessHoVerNetOutputD",
    "PostProcessHoVerNetOutputd",
]


class GenerateSuccinctContourd(MapTransform):
    """
    Args:
        height: height of bounding box, used to detect direction of line segment.
        width: width of bounding box, used to detect direction of line segment.
        allow_missing_keys: don't raise exception if key is missing.

    """
    backend = GenerateSuccinctContour.backend
    def __init__(
        self,
        keys: KeysCollection,
        height: int,
        width: int,
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = GenerateSuccinctContour(height=height, width=width)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])

        return d


class PostProcessHoVerNetOutputd(MapTransform):
    """
    Dictionary-based transform that post processing image tiles. It assumes network has three branches, with a segmentation branch that
    returns `np_pred`, a hover map branch that returns `hv_pred` and an optional classification branch that returns `nc_pred`. After this
    tranform, it will return pixel-wise nuclear instance segmentation prediction and a instance-level information dictionary.

    Args:
        hover_pred_key: hover map branch output key. Defaults to `HoVerNet.Branch.HV.value`.
        type_pred_key: classification branch output key. Defaults to `HoVerNet.Branch.NC.value`.
        inst_info_dict_key: a dict contaning a instance-level information dictionary will be added, which including bounding_box,
            centroid and contour. If output_classes is not None, the dictionary will also contain pixel-wise nuclear type prediction.
            Defaults to "inst_info".
        output_classes: number of types considered at output of NC branch. Defaults to None.
        return_centroids: whether to generate coords for each nucleus instance.
            Defaults to True.
        threshold_overall: threshold the float values of overall gradient map to int 0 or 1 with specified theashold.
            Defaults to 0.4.
        min_size: objects smaller than this size are removed. Defaults to 10.
        sigma: std. could be a single value, or `spatial_dims` number of values. Defaults to 0.4.
        kernel_size: the size of the Sobel kernel. Defaults to 21.
        radius: the radius of the disk-shaped footprint. Defaults to 2.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = PostProcessHoVerNetOutput.backend

    def __init__(
        self,
        keys: KeysCollection = HoVerNet.Branch.NP.value,
        hover_pred_key: str = HoVerNet.Branch.HV.value,
        type_pred_key: str = HoVerNet.Branch.NC.value,
        inst_info_dict_key: str = "inst_info",
        output_classes: Optional[int] = None,
        return_centroids: bool = True,
        threshold_overall: float = 0.4,
        min_size: int = 10,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.4,
        kernel_size: int = 21,
        radius: int = 2,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.hover_pred_key = hover_pred_key
        self.type_pred_key = type_pred_key
        self.inst_info_dict_key = inst_info_dict_key
        self.output_classes = output_classes
        self.return_centroids = return_centroids

        self.converter = PostProcessHoVerNetOutput(
            output_classes=output_classes,
            return_centroids=return_centroids,
            threshold_overall=threshold_overall,
            min_size=min_size,
            sigma=sigma,
            kernel_size=kernel_size,
            radius=radius,
        )

    def __call__(self, pred: Mapping[Hashable, NdarrayOrTensor]):
        """
        Args:
            pred: a dict combined output of classification(NC, optional), segmentation(NP) and hover map(HV) branches.

        Returns:
            pixel-wise nuclear instance segmentation prediction and a instance-level information dictionary stored in
            `inst_info_dict_key`.
        """
        d = dict(pred)
        for key in self.key_iterator(d):
            seg_pred = d[key]
            hover_pred = d[self.hover_pred_key]
            if self.output_classes is not None:
                type_pred = d[self.type_pred_key]
            else:
                type_pred = None

            d[key], inst_info_dict = self.converter(seg_pred, hover_pred, type_pred)
            d[self.inst_info_dict_key] = inst_info_dict

        return d


PostProcessHoVerNetOutputDict = PostProcessHoVerNetOutputD = PostProcessHoVerNetOutputd
GenerateSuccinctContourDict = GenerateSuccinctContourD = GenerateSuccinctContourd
