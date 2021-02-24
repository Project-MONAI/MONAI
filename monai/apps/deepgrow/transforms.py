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

from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch

from monai.config import IndexSelection, KeysCollection
from monai.networks.layers import GaussianFilter
from monai.transforms import SpatialCrop
from monai.transforms.transform import MapTransform, Randomizable, Transform
from monai.transforms.utils import generate_spatial_bounding_box
from monai.utils import min_version, optional_import

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)
distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")


# Transforms to support Training for Deepgrow models
class FindAllValidSlicesd(Transform):
    """
    Find/List all valid slices in the label.
    Label is assumed to be a 4D Volume with shape CDHW, where C=1.

    Args:
        label: key to the label source.
        sids: key to store slices indices having valid label map.
    """

    def __init__(self, label: str = "label", sids: str = "sids"):
        self.label = label
        self.sids = sids

    def _apply(self, label):
        sids = []
        for sid in range(label.shape[1]):  # Assume channel is first
            if np.sum(label[0][sid]) != 0:
                sids.append(sid)
        return np.asarray(sids)

    def __call__(self, data):
        d = dict(data)
        label = d[self.label]
        if label.shape[0] != 1:
            raise ValueError("Only supports single channel labels!")

        if len(label.shape) != 4:  # only for 3D
            raise ValueError("Only supports label with shape CDHW!")

        sids = self._apply(label)
        if sids is not None and len(sids):
            d[self.sids] = sids
        return d


class AddInitialSeedPointd(Randomizable, Transform):
    """
    Add random guidance as initial seed point for a given label.

    Note that the label is of size (C, D, H, W) or (C, H, W)
    The guidance is of size (2, N, # of dims) where N is number of guidance added
    # of dims = 4 when C, D, H, W; # of dims = 3 when (C, H, W)

    Args:
        label: label source.
        guidance: key to store guidance.
        sids: key that represents list of valid slice indices for the given label.
        sid: key that represents the slice to add initial seed point.  If not present, random sid will be chosen.
        connected_regions: maximum connected regions to use for adding initial points.
    """

    def __init__(
        self,
        label: str = "label",
        guidance: str = "guidance",
        sids: str = "sids",
        sid: str = "sid",
        connected_regions: int = 5,
    ):
        self.label = label
        self.sids = sids
        self.sid = sid
        self.guidance = guidance
        self.connected_regions = connected_regions

    def randomize(self, data=None):
        pass

    def _apply(self, label, sid):
        dimensions = 3 if len(label.shape) > 3 else 2
        default_guidance = [-1] * (dimensions + 1)

        dims = dimensions
        if sid is not None and dimensions == 3:
            dims = 2
            label = label[0][sid][np.newaxis]  # Assume channel is first

        label = (label > 0.5).astype(np.float32)
        blobs_labels = measure.label(label.astype(int), background=0) if dims == 2 else label
        assert np.max(blobs_labels) > 0, "Not a valid Label"

        pos_guidance = []
        for ridx in range(1, 2 if dims == 3 else self.connected_regions + 1):
            if dims == 2:
                label = (blobs_labels == ridx).astype(np.float32)
                if np.sum(label) == 0:
                    pos_guidance.append(default_guidance)
                    continue

            distance = distance_transform_cdt(label).flatten()
            probability = np.exp(distance) - 1.0

            idx = np.where(label.flatten() > 0)[0]
            seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
            dst = distance[seed]

            g = np.asarray(np.unravel_index(seed, label.shape)).transpose().tolist()[0]
            g[0] = dst[0]  # for debug
            if dimensions == 2 or dims == 3:
                pos_guidance.append(g)
            else:
                pos_guidance.append([g[0], sid, g[-2], g[-1]])

        return np.asarray([pos_guidance, [default_guidance] * len(pos_guidance)])

    def __call__(self, data):
        d = dict(data)
        sid = d.get(self.sid, None)
        sids = d.get(self.sids, None)
        if sids is not None:
            if sid is None or sid not in sids:
                sid = self.R.choice(sids, replace=False)
        else:
            sid = None
        d[self.guidance] = self._apply(d[self.label], sid)
        return d


class AddGuidanceSignald(Transform):
    """
    Add Guidance signal for input image.

    Based on the "guidance" points, apply gaussian to them and add them as new channel for input image.

    Args:
        image: key to the image source.
        guidance: key to store guidance.
        sigma: standard deviation for Gaussian kernel.
        number_intensity_ch: channel index.
        batched: whether input is batched or not.
    """

    def __init__(
        self,
        image: str = "image",
        guidance: str = "guidance",
        sigma: int = 2,
        number_intensity_ch: int = 1,
        batched: bool = False,
    ):
        self.image = image
        self.guidance = guidance
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch
        self.batched = batched

    def _get_signal(self, image, guidance):
        dimensions = 3 if len(image.shape) > 3 else 2
        guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        if dimensions == 3:
            signal = np.zeros((len(guidance), image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
        else:
            signal = np.zeros((len(guidance), image.shape[-2], image.shape[-1]), dtype=np.float32)

        sshape = signal.shape
        for i in range(len(guidance)):
            for point in guidance[i]:
                if np.any(np.asarray(point) < 0):
                    continue

                if dimensions == 3:
                    p1 = max(0, min(int(point[-3]), sshape[-3] - 1))
                    p2 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p3 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[i, p1, p2, p3] = 1.0
                else:
                    p1 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p2 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[i, p1, p2] = 1.0

            if np.max(signal[i]) > 0:
                signal_tensor = torch.tensor(signal[i])
                pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                signal_tensor = signal_tensor.squeeze(0).squeeze(0)
                signal[i] = signal_tensor.detach().cpu().numpy()
                signal[i] = (signal[i] - np.min(signal[i])) / (np.max(signal[i]) - np.min(signal[i]))
        return signal

    def _apply(self, image, guidance):
        if not self.batched:
            signal = self._get_signal(image, guidance)
            return np.concatenate([image, signal], axis=0)

        images = []
        for i, g in zip(image, guidance):
            i = i[0 : 0 + self.number_intensity_ch, ...]
            signal = self._get_signal(i, g)
            images.append(np.concatenate([i, signal], axis=0))
        return images

    def __call__(self, data):
        d = dict(data)
        image = d[self.image]
        guidance = d[self.guidance]

        d[self.image] = self._apply(image, guidance)
        return d


class FindDiscrepancyRegionsd(Transform):
    """
    Find discrepancy between prediction and actual during click interactions during training.

    If batched is true:
        label is in shape (B, C, D, H, W) or (B, C, H, W)
        pred has same shape as label
        discrepancy will have shape (B, 2, C, D, H, W) or (B, 2, C, H, W)

    Args:
        label: key to label source.
        pred: key to prediction source.
        discrepancy: key to store discrepancies found between label and prediction.
        batched: whether input is batched or not.
    """

    def __init__(
        self, label: str = "label", pred: str = "pred", discrepancy: str = "discrepancy", batched: bool = True
    ):
        self.label = label
        self.pred = pred
        self.discrepancy = discrepancy
        self.batched = batched

    @staticmethod
    def disparity(label, pred):
        label = (label > 0.5).astype(np.float32)
        pred = (pred > 0.5).astype(np.float32)
        disparity = label - pred

        pos_disparity = (disparity > 0).astype(np.float32)
        neg_disparity = (disparity < 0).astype(np.float32)
        return [pos_disparity, neg_disparity]

    def _apply(self, label, pred):
        if not self.batched:
            return self.disparity(label, pred)

        disparity = []
        for la, pr in zip(label, pred):
            disparity.append(self.disparity(la, pr))
        return disparity

    def __call__(self, data):
        d = dict(data)
        label = d[self.label]
        pred = d[self.pred]

        d[self.discrepancy] = self._apply(label, pred)
        return d


class AddRandomGuidanced(Randomizable, Transform):
    """
    Add random guidance based on discrepancies that were found between label and prediction.

    If batched is True:

        Guidance is of shape (B, 2, N, # of dim) where B is batch size, 2 means positive and negative,
        N means how many guidance points, # of dim is the total number of dimensions of the image
        (for example if the image is CDHW, then # of dim would be 4).

        Discrepancy is of shape (B, 2, C, D, H, W) or (B, 2, C, H, W)

        Probability is of shape (B,)

    Args:
        guidance: key to guidance source.
        discrepancy: key that represents discrepancies found between label and prediction.
        probability: key that represents click/interaction probability.
        batched: whether input is batched or not.
    """

    def __init__(
        self,
        guidance: str = "guidance",
        discrepancy: str = "discrepancy",
        probability: str = "probability",
        batched: bool = True,
    ):
        self.guidance = guidance
        self.discrepancy = discrepancy
        self.probability = probability
        self.batched = batched
        self._will_interact = None

    def randomize(self, data=None):
        probability = data[self.probability]
        if not self.batched:
            self._will_interact = self.R.choice([True, False], p=[probability, 1.0 - probability])
        else:
            self._will_interact = []
            for p in probability:
                self._will_interact.append(self.R.choice([True, False], p=[p, 1.0 - p]))

    def find_guidance(self, discrepancy):
        distance = distance_transform_cdt(discrepancy).flatten()
        probability = np.exp(distance) - 1.0
        idx = np.where(discrepancy.flatten() > 0)[0]

        if np.sum(discrepancy > 0) > 0:
            seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
            dst = distance[seed]

            g = np.asarray(np.unravel_index(seed, discrepancy.shape)).transpose().tolist()[0]
            g[0] = dst[0]
            return g
        return None

    def add_guidance(self, discrepancy, will_interact):
        if not will_interact:
            return None, None

        pos_discr = discrepancy[0]
        neg_discr = discrepancy[1]

        can_be_positive = np.sum(pos_discr) > 0
        can_be_negative = np.sum(neg_discr) > 0
        correct_pos = np.sum(pos_discr) >= np.sum(neg_discr)

        if correct_pos and can_be_positive:
            return self.find_guidance(pos_discr), None

        if not correct_pos and can_be_negative:
            return None, self.find_guidance(neg_discr)
        return None, None

    def _apply(self, guidance, discrepancy):
        guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        if not self.batched:
            pos, neg = self.add_guidance(discrepancy, self._will_interact)
            if pos:
                guidance[0].append(pos)
                guidance[1].append([-1] * len(pos))
            if neg:
                guidance[0].append([-1] * len(neg))
                guidance[1].append(neg)
        else:
            for g, d, w in zip(guidance, discrepancy, self._will_interact):
                pos, neg = self.add_guidance(d, w)
                if pos:
                    g[0].append(pos)
                    g[1].append([-1] * len(pos))
                if neg:
                    g[0].append([-1] * len(neg))
                    g[1].append(neg)
        return np.asarray(guidance)

    def __call__(self, data):
        d = dict(data)
        guidance = d[self.guidance]
        discrepancy = d[self.discrepancy]

        self.randomize(data)
        d[self.guidance] = self._apply(guidance, discrepancy)
        return d


class SpatialCropForegroundd(MapTransform):
    """
    Crop only the foreground object of the expected images.

    Difference VS CropForegroundd:

      1. If the bounding box is smaller than spatial size in all dimensions then this transform will crop the
         object using box's center and spatial_size.

      2. This transform will set "start_coord_key", "end_coord_key", "original_shape_key" and "cropped_shape_key"
         in data[{key}_{meta_key_postfix}]

    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    The valid part can be determined by any field in the data with `source_key`, for example:
    - Select values > 0 in image field as the foreground and crop on all fields specified by `keys`.
    - Select label = 3 in label field as the foreground to crop on all fields specified by `keys`.
    - Select label > 0 in the third channel of a One-Hot label field as the foreground to crop all `keys` fields.
    Users can define arbitrary function to select expected foreground from the whole source image or specified
    channels. And it can also add margin to every dim of the bounding box of foreground object.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.MapTransform`
        source_key: data source to generate the bounding box of foreground, can be image or label, etc.
        spatial_size: minimal spatial size of the image patch e.g. [128, 128, 128] to fit in.
        select_fn: function to select expected foreground, default is to select values > 0.
        channel_indices: if defined, select foreground only on the specified channels
            of image. if None, select foreground on the whole image.
        margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
        meta_key_postfix: use `{key}_{meta_key_postfix}` to to fetch/store the meta data according to the key data,
            default is `meta_dict`, the meta data is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        start_coord_key: key to record the start coordinate of spatial bounding box for foreground.
        end_coord_key: key to record the end coordinate of spatial bounding box for foreground.
        original_shape_key: key to record original shape for foreground.
        cropped_shape_key: key to record cropped shape for foreground.
    """

    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        spatial_size: Union[Sequence[int], np.ndarray],
        select_fn: Callable = lambda x: x > 0,
        channel_indices: Optional[IndexSelection] = None,
        margin: int = 0,
        meta_key_postfix="meta_dict",
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        original_shape_key: str = "foreground_original_shape",
        cropped_shape_key: str = "foreground_cropped_shape",
    ) -> None:
        super().__init__(keys)

        self.source_key = source_key
        self.spatial_size = list(spatial_size)
        self.select_fn = select_fn
        self.channel_indices = channel_indices
        self.margin = margin
        self.meta_key_postfix = meta_key_postfix
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.original_shape_key = original_shape_key
        self.cropped_shape_key = cropped_shape_key

    def __call__(self, data):
        d = dict(data)
        box_start, box_end = generate_spatial_bounding_box(
            d[self.source_key], self.select_fn, self.channel_indices, self.margin
        )

        center = np.mean([box_start, box_end], axis=0).astype(int).tolist()
        current_size = np.subtract(box_end, box_start).astype(int).tolist()

        if np.all(np.less(current_size, self.spatial_size)):
            cropper = SpatialCrop(roi_center=center, roi_size=self.spatial_size)
            box_start = cropper.roi_start
            box_end = cropper.roi_end
        else:
            cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)

        for key in self.keys:
            meta_key = f"{key}_{self.meta_key_postfix}"
            d[meta_key][self.start_coord_key] = box_start
            d[meta_key][self.end_coord_key] = box_end
            d[meta_key][self.original_shape_key] = d[key].shape

            image = cropper(d[key])
            d[meta_key][self.cropped_shape_key] = image.shape
            d[key] = image
        return d
