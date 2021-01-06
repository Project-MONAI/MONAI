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

from collections.abc import Sequence
from functools import partial
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.utils import eval_mode
from monai.visualize import default_normalizer, default_upsampler

try:
    from tqdm import trange

    trange = partial(trange, desc="Computing occlusion sensitivity")
except (ImportError, AttributeError):
    trange = range


def _check_input_image(image):
    """Check that the input image is as expected."""
    # Only accept batch size of 1
    if image.shape[0] > 1:
        raise RuntimeError("Expected batch size of 1.")


def _check_input_label(model, label, image):
    """Check that the input label is as expected."""
    if label is None:
        label = model(image).argmax(1)
    # If necessary turn the label into a 1-element tensor
    elif not isinstance(label, torch.Tensor):
        label = torch.tensor([[label]], dtype=torch.int64).to(image.device)
    # make sure there's only 1 element
    if label.numel() != image.shape[0]:
        raise RuntimeError("Expected as many labels as batches.")
    return label


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


def _append_to_sensitivity_im(model, batch_images, batch_ids, sensitivity_im):
    """For given number of images, get probability of predicting
    a given label. Append to previous evaluations."""
    batch_images = torch.cat(batch_images, dim=0)
    batch_ids = torch.LongTensor(batch_ids).unsqueeze(1).to(sensitivity_im.device)
    scores = model(batch_images).detach().gather(1, batch_ids)
    return torch.cat((sensitivity_im, scores))


class OcclusionSensitivity:
    """
    This class computes the occlusion sensitivity for a model's prediction
    of a given image. By occlusion sensitivity, we mean how the probability of a given
    prediction changes as the occluded section of an image changes. This can
    be useful to understand why a network is making certain decisions.

    The result is given as ``baseline`` (the probability of
    a certain output) minus the probability of the output with the occluded
    area.

    Therefore, higher values in the output image mean there was a
    greater the drop in certainty, indicating the occluded region was more
    important in the decision process.

    See: R. R. Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization. https://doi.org/10.1109/ICCV.2017.74

    Examples

    .. code-block:: python

        # densenet 2d
        from monai.networks.nets import densenet121
        from monai.visualize import OcclusionSensitivity

        model_2d = densenet121(spatial_dims=2, in_channels=1, out_channels=3)
        occ_sens = OcclusionSensitivity(nn_module=model_2d)
        result = occ_sens(x=torch.rand((1, 1, 48, 64)), class_idx=None, b_box=[-1, -1, 2, 40, 1, 62])

        # densenet 3d
        from monai.networks.nets import DenseNet
        from monai.visualize import OcclusionSensitivity

        model_3d = DenseNet(spatial_dims=3, in_channels=1, out_channels=3, init_features=2, growth_rate=2, block_config=(6,))
        occ_sens = OcclusionSensitivity(nn_module=model_3d, n_batch=10, stride=2)
        result = occ_sens(torch.rand(1, 1, 6, 6, 6), class_idx=1, b_box=[-1, -1, 2, 3, -1, -1, -1, -1])

    See Also:

        - :py:class:`monai.visualize.occlusion_sensitivity.OcclusionSensitivity.`
    """

    def __init__(
        self,
        nn_module: nn.Module,
        pad_val: float = 0.0,
        margin: Union[int, Sequence] = 2,
        n_batch: int = 128,
        stride: Union[int, Sequence] = 1,
        upsampler: Callable = default_upsampler,
        postprocessing: Callable = default_normalizer,
        verbose: bool = True,
    ) -> None:
        """Occlusion sensitivitiy constructor.

        :param nn_module: classification model to use for inference
        :param pad_val: when occluding part of the image, which values should we put
            in the image?
        :param margin: we'll create a cuboid/cube around the voxel to be occluded. if
            ``margin==2``, then we'll create a cube that is +/- 2 voxels in
            all directions (i.e., a cube of 5 x 5 x 5 voxels). A ``Sequence``
            can be supplied to have a margin of different sizes (i.e., create
            a cuboid).
        :param n_batch: number of images in a batch before inference.
        :param b_box: Bounding box on which to perform the analysis. The output image
            will also match in size. There should be a minimum and maximum for
            all dimensions except batch: ``[min1, max1, min2, max2,...]``.
            * By default, the whole image will be used. Decreasing the size will
            speed the analysis up, which might be useful for larger images.
            * Min and max are inclusive, so [0, 63, ...] will have size (64, ...).
            * Use -ve to use 0 for min values and im.shape[x]-1 for xth dimension.
        :param stride: Stride in spatial directions for performing occlusions. Can be single
            value or sequence (for varying stride in the different directions).
            Should be >= 1. Striding in the channel direction will always be 1.
        :param upsampler: An upsampling method to upsample the output image. Default is
            N dimensional linear (bilinear, trilinear, etc.) depending on num spatial
            dimensions of input.
        :param postprocessing: a callable that applies on the upsampled output image.
            default is normalising between 0 and 1.
        :param verbose: use ``tdqm.trange`` output (if available).
        """

        self.nn_module = nn_module
        self.upsampler = upsampler
        self.postprocessing = postprocessing
        self.pad_val = pad_val
        self.margin = margin
        self.n_batch = n_batch
        self.stride = stride
        self.verbose = verbose

    def _compute_occlusion_sensitivity(self, x, class_idx, b_box):

        # Get bounding box
        im_shape = np.array(x.shape[1:])
        b_box_min, b_box_max = _check_input_bounding_box(b_box, im_shape)

        # Get baseline probability
        baseline = self.nn_module(x).detach()[0, class_idx].item()

        # Create some lists
        batch_images = []
        batch_ids = []

        sensitivity_im = torch.empty(0, dtype=torch.float32, device=x.device)

        # If no bounding box supplied, output shape is same as input shape.
        # If bounding box is present, shape is max - min + 1
        output_im_shape = im_shape if b_box is None else b_box_max - b_box_min + 1

        # Calculate the downsampled shape
        if not isinstance(self.stride, Sequence):
            stride_np = np.full_like(im_shape, self.stride, dtype=np.int32)
            stride_np[0] = 1  # always do stride 1 in channel dimension
        else:
            # Convert to numpy array and check dimensions match
            stride_np = np.array(self.stride, dtype=np.int32)
            if stride_np.size != im_shape - 1:  # should be 1 less to get spatial dimensions
                raise ValueError(
                    "If supplying stride as sequence, number of elements of stride should match number of spatial dimensions."
                )

        # Obviously if stride = 1, downsampled_im_shape == output_im_shape
        downsampled_im_shape = np.floor(output_im_shape / stride_np).astype(np.int32)
        downsampled_im_shape[downsampled_im_shape == 0] = 1  # make sure dimension sizes are >= 1
        num_required_predictions = np.prod(downsampled_im_shape)

        # Loop 1D over image
        verbose_range = trange if self.verbose else range
        for i in verbose_range(num_required_predictions):
            # Get corresponding ND index
            idx = np.unravel_index(i, downsampled_im_shape)
            # Multiply by stride
            idx *= stride_np
            # If a bounding box is being used, we need to add on
            # the min to shift to start of region of interest
            if b_box_min is not None:
                idx += b_box_min

            # Get min and max index of box to occlude
            min_idx = [max(0, i - self.margin) for i in idx]
            max_idx = [min(j, i + self.margin) for i, j in zip(idx, im_shape)]

            # Clone and replace target area with `pad_val`
            occlu_im = x.detach().clone()
            occlu_im[(...,) + tuple(slice(i, j) for i, j in zip(min_idx, max_idx))] = self.pad_val

            # Add to list
            batch_images.append(occlu_im)
            batch_ids.append(class_idx)

            # Once the batch is complete (or on last iteration)
            if len(batch_images) == self.n_batch or i == num_required_predictions - 1:
                # Do the predictions and append to sensitivity map
                sensitivity_im = _append_to_sensitivity_im(self.nn_module, batch_images, batch_ids, sensitivity_im)
                # Clear lists
                batch_images = []
                batch_ids = []

        # Subtract baseline from sensitivity so that +ve values mean more important in decision process
        sensitivity_im = baseline - sensitivity_im

        # Reshape to match downsampled image, and unsqueeze to add batch dimension back in
        sensitivity_im = sensitivity_im.reshape(tuple(downsampled_im_shape)).unsqueeze(0)

        return sensitivity_im, output_im_shape

    def __call__(  # type: ignore
        self, x: torch.Tensor, class_idx: Optional[Union[int, torch.Tensor]] = None, b_box: Optional[Sequence] = None
    ):
        """
        Args:
            x: image to test. Should be tensor consisting of 1 batch, can be 2- or 3D.
            class_idx: classification label to check for changes. This could be the true
                label, or it could be the predicted label, etc. Use ``None`` to use generate
                the predicted model.
            b_box: Bounding box on which to perform the analysis. The output image
                will also match in size. There should be a minimum and maximum for
                all dimensions except batch: ``[min1, max1, min2, max2,...]``.
                * By default, the whole image will be used. Decreasing the size will
                speed the analysis up, which might be useful for larger images.
                * Min and max are inclusive, so [0, 63, ...] will have size (64, ...).
                * Use -ve to use 0 for min values and im.shape[x]-1 for xth dimension.
        Returns:
            Depends on the postprocessing, but the default return type is a Numpy array.
            The returned image will occupy the same space as the input image, unless a
            bounding box is supplied, in which case it will occupy that space. Unless
            upsampling is disabled, the output image will have voxels of the same size
            as the input image.
        """

        with eval_mode(self.nn_module):

            # Check input arguments
            _check_input_image(x)
            class_idx = _check_input_label(self.nn_module, class_idx, x)

            # Generate sensitivity image
            sensitivity_im, output_im_shape = self._compute_occlusion_sensitivity(x, class_idx, b_box)

            # upsampling and postprocessing
            if self.upsampler is not None:
                if np.any(output_im_shape != x.shape[1:]):
                    img_spatial = tuple(output_im_shape[1:])
                    sensitivity_im = self.upsampler(img_spatial)(sensitivity_im)
            if self.postprocessing:
                sensitivity_im = self.postprocessing(sensitivity_im)

            # Squeeze and return
            return sensitivity_im
