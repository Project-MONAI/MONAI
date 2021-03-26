# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import numpy as np
import torch

from monai.utils import ProbNMS


class PathologyProbNMS(ProbNMS):
    """
    This class extends monai.utils.ProbNMS and add the `resolution` option for
    Pathology.
    """

    def __call__(
        self,
        probs_map: Union[np.ndarray, torch.Tensor],
        resolution_level: int = 0,
    ):
        """
        probs_map: the input probabilities map, it must have shape (H[, W, ...]).
                resolution_level: the level at which the probabilities map is made.
        """
        if self.sigma != 0:
            if not isinstance(probs_map, torch.Tensor):
                probs_map = torch.as_tensor(probs_map, dtype=torch.float)
            self.filter.to(probs_map)
            probs_map = self.filter(probs_map)
        else:
            if not isinstance(probs_map, torch.Tensor):
                probs_map = probs_map.copy()

        if isinstance(probs_map, torch.Tensor):
            probs_map = probs_map.detach().cpu().numpy()

        probs_map_shape = probs_map.shape
        resolution = pow(2, resolution_level)

        outputs = []
        while np.max(probs_map) > self.prob_threshold:
            max_idx = np.unravel_index(probs_map.argmax(), probs_map_shape)
            prob_max = probs_map[max_idx]
            max_idx_arr = np.asarray(max_idx)
            coord_wsi = ((max_idx_arr + 0.5) * resolution).astype(int)
            outputs.append([prob_max] + list(coord_wsi))

            idx_min_range = (max_idx_arr - self.box_lower_bd).clip(0, None)
            idx_max_range = (max_idx_arr + self.box_upper_bd).clip(None, probs_map_shape)
            # for each dimension, set values during index ranges to 0
            slices = tuple([slice(idx_min_range[i], idx_max_range[i]) for i in range(self.spatial_dims)])
            probs_map[slices] = 0

        return outputs
