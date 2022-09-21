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

from typing import Hashable, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from monai.apps.pathology.transforms.post.array import PostProcessHoVerNetOutput
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.networks.nets import HoVerNet
from monai.transforms.transform import MapTransform
from monai.utils import optional_import

find_contours, has_findcontours = optional_import("skimage.measure", name="find_contours")
moments, has_moments = optional_import("skimage.measure", name="moments")

__all__ = ["PostProcessHoVerNetOutputDict", "PostProcessHoVerNetOutputD", "PostProcessHoVerNetOutputd"]


class PostProcessHoVerNetOutputd(MapTransform):
    """
    Post processing script for image tiles. It assumes that the output of the network is a dictionary, including segmentation,
    hovermap, and classification(optional) branch.
    """

    backend = PostProcessHoVerNetOutput.backend

    def __init__(
        self,
        keys: KeysCollection = HoVerNet.Branch.NP.value,
        HV_pred_key: str = HoVerNet.Branch.HV.value,
        NC_pred_key: str = HoVerNet.Branch.NC.value,
        inst_info_dict_key: str = "inst_info",
        output_classes: Optional[int] = None,
        return_centroids: bool = True,
        threshold_pred: float = 0.5,
        threshold_overall: float = 0.4,
        min_size: int = 10,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.4,
        kernel_size: int = 17,
        radius: int = 2,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            HV_pred_key: hover map branch output key. Defaults to `HoVerNet.Branch.HV.value`.
            NC_pred_key: classification branch output key. Defaults to `HoVerNet.Branch.NC.value`.
            inst_info_dict_key: a dict contaning a instance-level information dictionary will be added, which including bounding_box,
                centroid and contour. If output_classes is not None, the dictionary will also contain pixel-wise nuclear type prediction.
                Defaults to "inst_info".
            threshold_pred: threshold the float values of prediction to int 0 or 1 with specified theashold. Defaults to 0.5.
            threshold_overall: threshold the float values of overall gradient map to int 0 or 1 with specified theashold.
                Defaults to 0.4.
            min_size: objects smaller than this size are removed. Defaults to 10.
            sigma: std. could be a single value, or `spatial_dims` number of values. Defaults to 0.4.
            kernel_size: the size of the Sobel kernel. Defaults to 17.
            radius: the radius of the disk-shaped footprint. Defaults to 2.
            output_classes: number of types considered at output of NC branch.
            return_centroids: whether to generate coords for each nucleus instance.
                Defaults to True.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.NP_pred_key = keys
        self.HV_pred_key = HV_pred_key
        self.NC_pred_key = NC_pred_key
        self.inst_info_dict_key = inst_info_dict_key
        self.output_classes = output_classes
        self.return_centroids = return_centroids

        self.converter = PostProcessHoVerNetOutput(
            output_classes=output_classes,
            return_centroids=return_centroids,
            threshold_pred=threshold_pred,
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
            NP_pred = d[key]
            HV_pred = d[self.HV_pred_key]
            if self.output_classes is not None:
                NC_pred = d[self.NC_pred_key]
            else:
                NC_pred = None

            d[key], inst_info_dict = self.converter(NP_pred, HV_pred, NC_pred)
            d[self.inst_info_dict_key] = inst_info_dict

        return d


PostProcessHoVerNetOutputDict = PostProcessHoVerNetOutputD = PostProcessHoVerNetOutputd
