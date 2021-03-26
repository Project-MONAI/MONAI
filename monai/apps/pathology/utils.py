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
        resolution = pow(2, resolution_level)
        org_outputs = ProbNMS.__call__(self, probs_map)
        outputs = []
        for org_output in org_outputs:
            prob = org_output[0]
            coord = np.asarray(org_output[1:])
            coord_wsi = ((coord + 0.5) * resolution).astype(int)
            outputs.append([prob] + list(coord_wsi))
        return outputs
