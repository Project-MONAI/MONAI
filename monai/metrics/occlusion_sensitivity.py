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

from collections import Sequence
from typing import Union

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm import trange
except (ImportError, AttributeError):
    trange = range


def compute_occlusion_sensitivity(
    model: nn.Module,
    image: Union[np.ndarray, torch.Tensor],
    label: Union[int, torch.Tensor],
    pad_val: float = 0.0,
    margin: Union[int, Sequence] = 2,
    n_batch: int = 128,
) -> np.ndarray:
    """
    This function computes the occlusion sensitivity for a model's prediction
    of a given image. The result is given as ``baseline`` (the probability of
    a certain output) minus the probability of the output with the occluded
    area. Therefore, in the output image, higher values mean there was a
    greater the drop in certainty, indicating the occluded region was more
    important in the decision process.

    By occlusion sensitivity, we mean how the probability of a given
    prediction changes as the occluded section of an image changes. This can
    be useful to understand why a network is making certain decisions.

    Args:
        model: model to use for inference
        image: image to test
        label: classification label to check for changes (normally the true
            label, but doesn't have to be)
        pad_val: when occluding part of the image, which values should we put
            in the image?
        margin: we'll create a cuboid/cube around the voxel to be occluded. if
            ``margin==2``, then we'll create a cube that is +/- 2 voxels in
            all directions (i.e., a cube of 5 x 5 x 5 voxels). A ``Sequence``
            can be supplied to have a margin of different sizes (i.e., create
            a cuboid).
        n_batch: number of images in a batch before inference.
    """

    # If necessary turn the label into a 1-element tensor
    if isinstance(label, int):
        label = torch.tensor([[label]], dtype=torch.int64).to(image.device)

    # Get image shape
    im_shape = image.shape[1:]

    # Get baseline probability
    baseline = model(image).detach()[0, label].item()

    # Create some lists
    batch_images_lst = []
    batch_ids_lst = []

    heatmap = torch.empty(0, dtype=torch.float32, device=image.device)
    # Loop 1D over image
    for i in trange(image.numel()):
        # Get corresponding ND index
        idx = np.unravel_index(i, im_shape)
        # Get min and max index of bounding box.
        min_idx = [max(0, i - margin) for i in idx]
        max_idx = [min(j, i + margin) for i, j in zip(idx, im_shape)]

        # Clone and replace target area with `pad_val`
        occlu_im = image.clone()
        occlu_im[..., min_idx[0] : max_idx[0], min_idx[1] : max_idx[1], min_idx[2] : max_idx[2]] = pad_val

        # Add to list
        batch_images_lst.append(occlu_im)
        batch_ids_lst.append(label)

        # Once the batch is complete (or on last iteration)
        if len(batch_images_lst) == n_batch or i == image.numel() - 1:
            # Get the predictions and append to tensor
            batch_images = torch.cat(batch_images_lst, dim=0)
            batch_ids = torch.cat(batch_ids_lst, dim=0)
            scores = model(batch_images).detach().gather(1, batch_ids)
            heatmap = torch.cat((heatmap, scores))

            # Clear lists
            batch_images_lst = []
            batch_ids_lst = []

    # Convert tensor to numpy and reshape
    diffmaps = np.squeeze(heatmap.cpu().numpy().reshape(im_shape))

    # Subtract from baseline and return
    return baseline - diffmaps
