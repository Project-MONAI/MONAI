# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from .utils import sliding_window_inference
import torch


class Inferer(ABC):
    """
    A base class for model inference.
    Extend this class to support operations during inference, e.g. a sliding window method.
    """

    @abstractmethod
    def __call__(self, inputs: torch.Tensor, network):
        """
        Run inference on `inputs` with the `network` model.

        Args:
            inputs (torch.tensor): input of the model inference.
            network (Network): model for inference.
        """
        raise NotImplementedError("subclass will implement the operations.")


class SimpleInferer(Inferer):
    """
    SimpleInferer is the normal inference method that run model forward() directly.

    """

    def __init__(self):
        Inferer.__init__(self)

    def __call__(self, inputs: torch.Tensor, network):
        """Unified callable function API of Inferers.

        Args:
            inputs (torch.tensor): model input data for inference.
            network (Network): target model to execute inference.

        """
        return network(inputs)


class SlidingWindowInferer(Inferer):
    """
    Sliding window method for model inference,
    with `sw_batch_size` windows for every model.forward().

    Args:
        roi_size (list, tuple): the window size to execute SlidingWindow evaluation.
        sw_batch_size (int): the batch size to run window slices.
        overlap (float): Amount of overlap between scans.
        blend_mode (str): How to blend output of overlapping windows. Options are 'constant', 'gaussian'. 'constant'
            gives equal weight to all predictions while gaussian gives less weight to predictions on edges of windows.

    Note:
        the "sw_batch_size" here is to run a batch of window slices of 1 input image,
        not batch size of input images.

    """

    def __init__(self, roi_size, sw_batch_size: int = 1, overlap: float = 0.25, blend_mode: str = "constant"):
        Inferer.__init__(self)
        if not isinstance(roi_size, (list, tuple)):
            raise ValueError("must specify the roi size in a list or tuple for SlidingWindow.")
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.blend_mode = blend_mode

    def __call__(self, inputs: torch.Tensor, network):
        """
        Unified callable function API of Inferers.

        Args:
            inputs (torch.tensor): model input data for inference.
            network (Network): target model to execute inference.

        """
        return sliding_window_inference(
            inputs, self.roi_size, self.sw_batch_size, network, self.overlap, self.blend_mode
        )
