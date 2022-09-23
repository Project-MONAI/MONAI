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

from typing import Dict, Hashable, Mapping, Sequence, Union

import torch

from monai.apps.pathology.transforms.post.array import CalcualteInstanceSegmentationMap
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms.transform import MapTransform

__all__ = [
    "CalcualteInstanceSegmentationMapD",
    "CalcualteInstanceSegmentationMapDict",
    "CalcualteInstanceSegmentationMapd",
]


class CalcualteInstanceSegmentationMapd(MapTransform):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        keys: keys of the corresponding items to model output.
        hover_keys: keys of the probability map from nucleus prediction branch.
        threshold_pred: threshold the float values of prediction to int 0 or 1 with specified theashold. Defaults to 0.5.
        threshold_overall: threshold the float values of overall gradient map to int 0 or 1 with specified theashold.
            Defaults to 0.4.
        min_size: objects smaller than this size are removed. Defaults to 10.
        sigma: std. could be a single value, or `spatial_dims` number of values. Defaults to 0.4.
        kernel_size: the size of the Sobel kernel. Defaults to 17.
        radius: the radius of the disk-shaped footprint. Defaults to 2.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = CalcualteInstanceSegmentationMap.backend

    def __init__(
        self,
        keys: KeysCollection,
        hover_key: str,
        threshold_pred: float = 0.5,
        threshold_overall: float = 0.4,
        min_size: int = 10,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.4,
        kernel_size: int = 17,
        radius: int = 2,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.hover_key = hover_key
        self.transform = CalcualteInstanceSegmentationMap(
            threshold_pred=threshold_pred,
            threshold_overall=threshold_overall,
            min_size=min_size,
            sigma=sigma,
            kernel_size=kernel_size,
            radius=radius,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key], d[self.hover_key])

        return d  # type: ignore


CalcualteInstanceSegmentationMapD = CalcualteInstanceSegmentationMapDict = CalcualteInstanceSegmentationMapd
