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

from collections.abc import Sequence
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.utils import eval_mode
from monai.visualize.visualizer import default_upsampler

try:
    from tqdm import trange

    trange = partial(trange, desc="Computing occlusion sensitivity")
except (ImportError, AttributeError):
    trange = range

# For stride two (for example),
# if input array is:     |0|1|2|3|4|5|6|7|
# downsampled output is: | 0 | 1 | 2 | 3 |
# So the upsampling should do it by the corners of the image, not their centres
default_upsampler = partial(default_upsampler, align_corners=True)


def _check_input_image(image):
    """Check that the input image is as expected."""
    # Only accept batch size of 1
    if image.shape[0] > 1:
        raise RuntimeError("Expected batch size of 1.")


def _check_input_bounding_box(b_box, im_shape):
    """Check that the bounding box (if supplied) is as expected."""
    # If no bounding box has been supplied, set min and max to None
    if b_box is None:
        b_box_min = b_box_max = None

    # Bounding box has been supplied
    else:
        # Should be twice as many elements in `b_box` as `im_shape`
        if len(b_box) != 2 * len(im_shape):
            raise ValueError("Bounding box should contain upper and lower for all dimensions (except batch number)")

        # If any min's or max's are -ve, set them to 0 and im_shape-1, respectively.
        b_box_min = np.array(b_box[::2])
        b_box_max = np.array(b_box[1::2])
        b_box_min[b_box_min < 0] = 0
        b_box_max[b_box_max < 0] = im_shape[b_box_max < 0] - 1
        # Check all max's are < im_shape
        if np.any(b_box_max >= im_shape):
            raise ValueError("Max bounding box should be < image size for all values")
        # Check all min's are <= max's
        if np.any(b_box_min > b_box_max):
            raise ValueError("Min bounding box should be <= max for all values")

    return b_box_min, b_box_max


def _append_to_sensitivity_ims(model, batch_images, sensitivity_ims):
    """Infer given images. Append to previous evaluations. Store each class separately."""
    batch_images = torch.cat(batch_images, dim=0)
    scores = model(batch_images).detach()
    for i in range(scores.shape[1]):
        sensitivity_ims[i] = torch.cat((sensitivity_ims[i], scores[:, i]))
    return sensitivity_ims


def _get_as_np_array(val, numel):
    # If not a sequence, then convert scalar to numpy array
    if not isinstance(val, Sequence):
        out = np.full(numel, val, dtype=np.int32)
        out[0] = 1  # mask_size and stride always 1 in channel dimension
    else:
        # Convert to numpy array and check dimensions match
        out = np.array(val, dtype=np.int32)
        # Add stride of 1 to the channel direction (since user input was only for spatial dimensions)
        out = np.insert(out, 0, 1)
        if out.size != numel:
            raise ValueError(
                "If supplying stride/mask_size as sequence, number of elements should match number of spatial dimensions."
            )
    return out


class OcclusionSensitivity:
    """
    This class computes the occlusion sensitivity for a model's prediction of a given image. By occlusion sensitivity,
    we mean how the probability of a given prediction changes as the occluded section of an image changes. This can be
    useful to understand why a network is making certain decisions.

    As important parts of the image are occluded, the probability of classifying the image correctly will decrease.
    Hence, more negative values imply the corresponding occluded volume was more important in the decision process.

    Two ``torch.Tensor`` will be returned by the ``__call__`` method: an occlusion map and an image of the most probable
    class. Both images will be cropped if a bounding box used, but voxel sizes will always match the input.

    The occlusion map shows the inference probabilities when the corresponding part of the image is occluded. Hence,
    more -ve values imply that region was important in the decision process. The map will have shape ``BCHW(D)N``,
    where ``N`` is the number of classes to be inferred by the network. Hence, the occlusion for class ``i`` can
    be seen with ``map[...,i]``.

    The most probable class is an image of the probable class when the corresponding part of the image is occluded
    (equivalent to ``occ_map.argmax(dim=-1)``).

    See: R. R. Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization. https://doi.org/10.1109/ICCV.2017.74.

    Examples:

    .. code-block:: python

        # densenet 2d
        from monai.networks.nets import DenseNet121
        from monai.visualize import OcclusionSensitivity

        model_2d = DenseNet121(spatial_dims=2, in_channels=1, out_channels=3)
        occ_sens = OcclusionSensitivity(nn_module=model_2d)
        occ_map, most_probable_class = occ_sens(x=torch.rand((1, 1, 48, 64)), class_idx=None, b_box=[-1, -1, 2, 40, 1, 62])

        # densenet 3d
        from monai.networks.nets import DenseNet
        from monai.visualize import OcclusionSensitivity

        model_3d = DenseNet(spatial_dims=3, in_channels=1, out_channels=3, init_features=2, growth_rate=2, block_config=(6,))
        occ_sens = OcclusionSensitivity(nn_module=model_3d, n_batch=10, stride=2)
        occ_map, most_probable_class = occ_sens(torch.rand(1, 1, 6, 6, 6), class_idx=1, b_box=[-1, -1, 2, 3, -1, -1, -1, -1])

    See Also:

        - :py:class:`monai.visualize.occlusion_sensitivity.OcclusionSensitivity.`
    """

    def __init__(
        self,
        nn_module: nn.Module,
        pad_val: Optional[float] = None,
        mask_size: Union[int, Sequence] = 15,
        n_batch: int = 128,
        stride: Union[int, Sequence] = 1,
        upsampler: Optional[Callable] = default_upsampler,
        verbose: bool = True,
    ) -> None:
        """Occlusion sensitivity constructor.

        Args:
            nn_module: Classification model to use for inference
            pad_val: When occluding part of the image, which values should we put
                in the image? If ``None`` is used, then the average of the image will be used.
            mask_size: Size of box to be occluded, centred on the central voxel. To ensure that the occluded area
                is correctly centred, ``mask_size`` and ``stride`` should both be odd or even.
            n_batch: Number of images in a batch for inference.
            stride: Stride in spatial directions for performing occlusions. Can be single
                value or sequence (for varying stride in the different directions).
                Should be >= 1. Striding in the channel direction will always be 1.
            upsampler: An upsampling method to upsample the output image. Default is
                N-dimensional linear (bilinear, trilinear, etc.) depending on num spatial
                dimensions of input.
            verbose: Use ``tdqm.trange`` output (if available).
        """

        self.nn_module = nn_module
        self.upsampler = upsampler
        self.pad_val = pad_val
        self.mask_size = mask_size
        self.n_batch = n_batch
        self.stride = stride
        self.verbose = verbose

    def _compute_occlusion_sensitivity(self, x, b_box):

        # Get bounding box
        im_shape = np.array(x.shape[1:])
        b_box_min, b_box_max = _check_input_bounding_box(b_box, im_shape)

        # Get the number of prediction classes
        num_classes = self.nn_module(x).numel()

        #  If pad val not supplied, get the mean of the image
        pad_val = x.mean() if self.pad_val is None else self.pad_val

        # List containing a batch of images to be inferred
        batch_images = []

        # List of sensitivity images, one for each inferred class
        sensitivity_ims = num_classes * [torch.empty(0, dtype=torch.float32, device=x.device)]

        # If no bounding box supplied, output shape is same as input shape.
        # If bounding box is present, shape is max - min + 1
        output_im_shape = im_shape if b_box is None else b_box_max - b_box_min + 1

        # Get the stride and mask_size as numpy arrays
        self.stride = _get_as_np_array(self.stride, len(im_shape))
        self.mask_size = _get_as_np_array(self.mask_size, len(im_shape))

        # For each dimension, ...
        for o, s in zip(output_im_shape, self.stride):
            # if the size is > 1, then check that the stride is a factor of the output image shape
            if o > 1 and o % s != 0:
                raise ValueError(
                    "Stride should be a factor of the image shape. Im shape "
                    + f"(taking bounding box into account): {output_im_shape}, stride: {self.stride}"
                )

        # to ensure the occluded area is nicely centred if stride is even, ensure that so is the mask_size
        if np.any(self.mask_size % 2 != self.stride % 2):
            raise ValueError(
                "Stride and mask size should both be odd or even (element-wise). "
                + f"``stride={self.stride}``, ``mask_size={self.mask_size}``"
            )

        downsampled_im_shape = (output_im_shape / self.stride).astype(np.int32)
        downsampled_im_shape[downsampled_im_shape == 0] = 1  # make sure dimension sizes are >= 1
        num_required_predictions = np.prod(downsampled_im_shape)

        # Get bottom left and top right corners of occluded region
        lower_corner = (self.stride - self.mask_size) // 2
        upper_corner = (self.stride + self.mask_size) // 2

        # Loop 1D over image
        verbose_range = trange if self.verbose else range
        for i in verbose_range(num_required_predictions):
            # Get corresponding ND index
            idx = np.unravel_index(i, downsampled_im_shape)
            # Multiply by stride
            idx *= self.stride
            # If a bounding box is being used, we need to add on
            # the min to shift to start of region of interest
            if b_box_min is not None:
                idx += b_box_min

            # Get min and max index of box to occlude (and make sure it's in bounds)
            min_idx = np.maximum(idx + lower_corner, 0)
            max_idx = np.minimum(idx + upper_corner, im_shape)

            # Clone and replace target area with `pad_val`
            occlu_im = x.detach().clone()
            occlu_im[(...,) + tuple(slice(i, j) for i, j in zip(min_idx, max_idx))] = pad_val

            # Add to list
            batch_images.append(occlu_im)

            # Once the batch is complete (or on last iteration)
            if len(batch_images) == self.n_batch or i == num_required_predictions - 1:
                # Do the predictions and append to sensitivity maps
                sensitivity_ims = _append_to_sensitivity_ims(self.nn_module, batch_images, sensitivity_ims)
                # Clear lists
                batch_images = []

        # Reshape to match downsampled image, and unsqueeze to add batch dimension back in
        for i in range(num_classes):
            sensitivity_ims[i] = sensitivity_ims[i].reshape(tuple(downsampled_im_shape)).unsqueeze(0)

        return sensitivity_ims, output_im_shape

    def __call__(  # type: ignore
        self,
        x: torch.Tensor,
        b_box: Optional[Sequence] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Image to use for inference. Should be a tensor consisting of 1 batch.
            b_box: Bounding box on which to perform the analysis. The output image will be limited to this size.
                There should be a minimum and maximum for all dimensions except batch: ``[min1, max1, min2, max2,...]``.
                * By default, the whole image will be used. Decreasing the size will speed the analysis up, which might
                    be useful for larger images.
                * Min and max are inclusive, so ``[0, 63, ...]`` will have size ``(64, ...)``.
                * Use -ve to use ``min=0`` and ``max=im.shape[x]-1`` for xth dimension.

        Returns:
            * Occlusion map:
                * Shows the inference probabilities when the corresponding part of the image is occluded.
                    Hence, more -ve values imply that region was important in the decision process.
                * The map will have shape ``BCHW(D)N``, where N is the number of classes to be inferred by the
                    network. Hence, the occlusion for class ``i`` can be seen with ``map[...,i]``.
            * Most probable class:
                * The most probable class when the corresponding part of the image is occluded (``argmax(dim=-1)``).
            Both images will be cropped if a bounding box used, but voxel sizes will always match the input.
        """

        with eval_mode(self.nn_module):

            # Check input arguments
            _check_input_image(x)

            # Generate sensitivity images
            sensitivity_ims_list, output_im_shape = self._compute_occlusion_sensitivity(x, b_box)

            # Loop over image for each classification
            for i in range(len(sensitivity_ims_list)):

                # upsample
                if self.upsampler is not None:
                    if len(sensitivity_ims_list[i].shape) != len(x.shape):
                        raise AssertionError
                    if np.any(sensitivity_ims_list[i].shape != x.shape):
                        img_spatial = tuple(output_im_shape[1:])
                        sensitivity_ims_list[i] = self.upsampler(img_spatial)(sensitivity_ims_list[i])

            # Convert list of tensors to tensor
            sensitivity_ims = torch.stack(sensitivity_ims_list, dim=-1)

            # The most probable class is the max in the classification dimension (last)
            most_probable_class = sensitivity_ims.argmax(dim=-1)

            return sensitivity_ims, most_probable_class
