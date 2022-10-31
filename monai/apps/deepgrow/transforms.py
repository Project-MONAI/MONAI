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
import json
from typing import Callable, Dict, Hashable, Optional, Sequence, Union

import numpy as np
import torch

from monai.config import IndexSelection, KeysCollection
from monai.networks.layers import GaussianFilter
from monai.transforms import Resize, SpatialCrop
from monai.transforms.transform import MapTransform, Randomizable, Transform
from monai.transforms.utils import generate_spatial_bounding_box, is_positive
from monai.utils import InterpolateMode, deprecated_arg, ensure_tuple, ensure_tuple_rep, min_version, optional_import
from monai.utils.enums import PostFix

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)
distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")

DEFAULT_POST_FIX = PostFix.meta()


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
        d: Dict = dict(data)
        label = d[self.label].numpy() if isinstance(data[self.label], torch.Tensor) else data[self.label]
        if label.shape[0] != 1:
            raise ValueError(f"Only supports single channel labels, got label shape {label.shape}!")

        if len(label.shape) != 4:  # only for 3D
            raise ValueError(f"Only supports label with shape CDHW, got label shape {label.shape}!")

        sids = self._apply(label)
        if sids is not None and len(sids):
            d[self.sids] = sids
        return d


class AddInitialSeedPointd(Randomizable, Transform):
    """
    Add random guidance as initial seed point for a given label.

    Note that the label is of size (C, D, H, W) or (C, H, W)

    The guidance is of size (2, N, # of dims) where N is number of guidance added.
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
        self.sids_key = sids
        self.sid_key = sid
        self.sid = None
        self.guidance = guidance
        self.connected_regions = connected_regions

    def randomize(self, data):
        sid = data.get(self.sid_key, None)
        sids = data.get(self.sids_key, None)
        if sids is not None:
            if sid is None or sid not in sids:
                sid = self.R.choice(sids, replace=False)
        else:
            sid = None
        self.sid = sid

    def _apply(self, label, sid):
        dimensions = 3 if len(label.shape) > 3 else 2
        default_guidance = [-1] * (dimensions + 1)

        dims = dimensions
        if sid is not None and dimensions == 3:
            dims = 2
            label = label[0][sid][np.newaxis]  # Assume channel is first

        label = (label > 0.5).astype(np.float32)
        blobs_labels = measure.label(label.astype(int), background=0) if dims == 2 else label
        if np.max(blobs_labels) <= 0:
            raise AssertionError("Not a valid Label")

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
        self.randomize(data)
        d[self.guidance] = json.dumps(self._apply(d[self.label], self.sid).astype(int, copy=False).tolist())
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

    """

    def __init__(self, image: str = "image", guidance: str = "guidance", sigma: int = 2, number_intensity_ch: int = 1):
        self.image = image
        self.guidance = guidance
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch

    def _get_signal(self, image, guidance):
        dimensions = 3 if len(image.shape) > 3 else 2
        guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        guidance = json.loads(guidance) if isinstance(guidance, str) else guidance
        if dimensions == 3:
            signal = np.zeros((len(guidance), image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
        else:
            signal = np.zeros((len(guidance), image.shape[-2], image.shape[-1]), dtype=np.float32)

        sshape = signal.shape
        for i, g_i in enumerate(guidance):
            for point in g_i:
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
        signal = self._get_signal(image, guidance)

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        image = image[0 : 0 + self.number_intensity_ch, ...]
        return np.concatenate([image, signal], axis=0)

    def __call__(self, data):
        d = dict(data)
        image = d[self.image]
        guidance = d[self.guidance]

        d[self.image] = self._apply(image, guidance)
        return d


class FindDiscrepancyRegionsd(Transform):
    """
    Find discrepancy between prediction and actual during click interactions during training.

    Args:
        label: key to label source.
        pred: key to prediction source.
        discrepancy: key to store discrepancies found between label and prediction.

    """

    def __init__(self, label: str = "label", pred: str = "pred", discrepancy: str = "discrepancy"):
        self.label = label
        self.pred = pred
        self.discrepancy = discrepancy

    @staticmethod
    def disparity(label, pred):
        label = (label > 0.5).astype(np.float32)
        pred = (pred > 0.5).astype(np.float32)
        disparity = label - pred

        pos_disparity = (disparity > 0).astype(np.float32)
        neg_disparity = (disparity < 0).astype(np.float32)
        return [pos_disparity, neg_disparity]

    def _apply(self, label, pred):
        return self.disparity(label, pred)

    def __call__(self, data):
        d = dict(data)
        label = d[self.label]
        pred = d[self.pred]

        d[self.discrepancy] = self._apply(label, pred)
        return d


class AddRandomGuidanced(Randomizable, Transform):
    """
    Add random guidance based on discrepancies that were found between label and prediction.
    input shape is as below:
    Guidance is of shape (2, N, # of dim)
    Discrepancy is of shape (2, C, D, H, W) or (2, C, H, W)
    Probability is of shape (1)

    Args:
        guidance: key to guidance source.
        discrepancy: key that represents discrepancies found between label and prediction.
        probability: key that represents click/interaction probability.

    """

    def __init__(self, guidance: str = "guidance", discrepancy: str = "discrepancy", probability: str = "probability"):
        self.guidance = guidance
        self.discrepancy = discrepancy
        self.probability = probability
        self._will_interact = None

    def randomize(self, data=None):
        probability = data[self.probability]
        self._will_interact = self.R.choice([True, False], p=[probability, 1.0 - probability])

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
        guidance = json.loads(guidance) if isinstance(guidance, str) else guidance
        pos, neg = self.add_guidance(discrepancy, self._will_interact)
        if pos:
            guidance[0].append(pos)
            guidance[1].append([-1] * len(pos))
        if neg:
            guidance[0].append([-1] * len(neg))
            guidance[1].append(neg)

        return json.dumps(np.asarray(guidance, dtype=int).tolist())

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

    Difference VS :py:class:`monai.transforms.CropForegroundd`:

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
        allow_smaller: when computing box size with `margin`, whether allow the image size to be smaller
            than box size, default to `True`. if the margined size is bigger than image size, will pad with
            specified `mode`.
        meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_keys is None, use `{key}_{meta_key_postfix}` to fetch/store the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        start_coord_key: key to record the start coordinate of spatial bounding box for foreground.
        end_coord_key: key to record the end coordinate of spatial bounding box for foreground.
        original_shape_key: key to record original shape for foreground.
        cropped_shape_key: key to record cropped shape for foreground.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        spatial_size: Union[Sequence[int], np.ndarray],
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        margin: int = 0,
        allow_smaller: bool = True,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix=DEFAULT_POST_FIX,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        original_shape_key: str = "foreground_original_shape",
        cropped_shape_key: str = "foreground_cropped_shape",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.source_key = source_key
        self.spatial_size = list(spatial_size)
        self.select_fn = select_fn
        self.channel_indices = channel_indices
        self.margin = margin
        self.allow_smaller = allow_smaller
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.original_shape_key = original_shape_key
        self.cropped_shape_key = cropped_shape_key

    def __call__(self, data):
        d = dict(data)
        box_start, box_end = generate_spatial_bounding_box(
            d[self.source_key], self.select_fn, self.channel_indices, self.margin, self.allow_smaller
        )

        center = list(np.mean([box_start, box_end], axis=0).astype(int, copy=False))
        current_size = list(np.subtract(box_end, box_start).astype(int, copy=False))

        if np.all(np.less(current_size, self.spatial_size)):
            cropper = SpatialCrop(roi_center=center, roi_size=self.spatial_size)
            box_start = np.array([s.start for s in cropper.slices])
            box_end = np.array([s.stop for s in cropper.slices])
        else:
            cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)

        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            meta_key = meta_key or f"{key}_{meta_key_postfix}"
            d[meta_key][self.start_coord_key] = box_start
            d[meta_key][self.end_coord_key] = box_end
            d[meta_key][self.original_shape_key] = d[key].shape

            image = cropper(d[key])
            d[meta_key][self.cropped_shape_key] = image.shape
            d[key] = image
        return d


# Transforms to support Inference for Deepgrow models
class AddGuidanceFromPointsd(Transform):
    """
    Add guidance based on user clicks.

    We assume the input is loaded by LoadImaged and has the shape of (H, W, D) originally.
    Clicks always specify the coordinates in (H, W, D)

    If depth_first is True:

        Input is now of shape (D, H, W), will return guidance that specifies the coordinates in (D, H, W)

    else:

        Input is now of shape (H, W, D), will return guidance that specifies the coordinates in (H, W, D)

    Args:
        ref_image: key to reference image to fetch current and original image details.
        guidance: output key to store guidance.
        foreground: key that represents user foreground (+ve) clicks.
        background: key that represents user background (-ve) clicks.
        axis: axis that represents slices in 3D volume. (axis to Depth)
        depth_first: if depth (slices) is positioned at first dimension.
        spatial_dims: dimensions based on model used for deepgrow (2D vs 3D).
        slice_key: key that represents applicable slice to add guidance.
        meta_keys: explicitly indicate the key of the metadata dictionary of `ref_image`.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            if None, will try to construct meta_keys by `{ref_image}_{meta_key_postfix}`.
        meta_key_postfix: if meta_key is None, use `{ref_image}_{meta_key_postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.

    .. deprecated:: 0.6.0
        ``dimensions`` is deprecated, use ``spatial_dims`` instead.

    """

    @deprecated_arg(name="dimensions", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        ref_image,
        guidance: str = "guidance",
        foreground: str = "foreground",
        background: str = "background",
        axis: int = 0,
        depth_first: bool = True,
        spatial_dims: int = 2,
        slice_key: str = "slice",
        meta_keys: Optional[str] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        dimensions: Optional[int] = None,
    ):
        self.ref_image = ref_image
        self.guidance = guidance
        self.foreground = foreground
        self.background = background
        self.axis = axis
        self.depth_first = depth_first
        self.dimensions = spatial_dims if dimensions is None else dimensions
        self.slice = slice_key
        self.meta_keys = meta_keys
        self.meta_key_postfix = meta_key_postfix

    def _apply(self, pos_clicks, neg_clicks, factor, slice_num):
        pos = neg = []

        if self.dimensions == 2:
            points = list(pos_clicks)
            points.extend(neg_clicks)
            points = np.array(points)

            slices = list(np.unique(points[:, self.axis]))
            slice_idx = slices[0] if slice_num is None else next(x for x in slices if x == slice_num)

            if len(pos_clicks):
                pos_clicks = np.array(pos_clicks)
                pos = (pos_clicks[np.where(pos_clicks[:, self.axis] == slice_idx)] * factor)[:, 1:].astype(int).tolist()
            if len(neg_clicks):
                neg_clicks = np.array(neg_clicks)
                neg = (neg_clicks[np.where(neg_clicks[:, self.axis] == slice_idx)] * factor)[:, 1:].astype(int).tolist()

            guidance = [pos, neg, slice_idx]
        else:
            if len(pos_clicks):
                pos = np.multiply(pos_clicks, factor).astype(int, copy=False).tolist()
            if len(neg_clicks):
                neg = np.multiply(neg_clicks, factor).astype(int, copy=False).tolist()
            guidance = [pos, neg]
        return guidance

    def __call__(self, data):
        d = dict(data)
        meta_dict_key = self.meta_keys or f"{self.ref_image}_{self.meta_key_postfix}"
        if meta_dict_key not in d:
            raise RuntimeError(f"Missing meta_dict {meta_dict_key} in data!")
        if "spatial_shape" not in d[meta_dict_key]:
            raise RuntimeError('Missing "spatial_shape" in meta_dict!')
        original_shape = d[meta_dict_key]["spatial_shape"]
        current_shape = list(d[self.ref_image].shape)

        if self.depth_first:
            if self.axis != 0:
                raise RuntimeError("Depth first means the depth axis should be 0.")
            # in here we assume the depth dimension was in the last dimension of "original_shape"
            original_shape = np.roll(original_shape, 1)

        factor = np.array(current_shape) / original_shape

        fg_bg_clicks = []
        for key in [self.foreground, self.background]:
            clicks = d[key]
            clicks = list(np.array(clicks, dtype=int))
            if self.depth_first:
                for i in range(len(clicks)):
                    clicks[i] = list(np.roll(clicks[i], 1))
            fg_bg_clicks.append(clicks)
        d[self.guidance] = self._apply(fg_bg_clicks[0], fg_bg_clicks[1], factor, d.get(self.slice))
        return d


class SpatialCropGuidanced(MapTransform):
    """
    Crop image based on guidance with minimal spatial size.

    - If the bounding box is smaller than spatial size in all dimensions then this transform will crop the
      object using box's center and spatial_size.

    - This transform will set "start_coord_key", "end_coord_key", "original_shape_key" and "cropped_shape_key"
      in data[{key}_{meta_key_postfix}]

    Input data is of shape (C, spatial_1, [spatial_2, ...])

    Args:
        keys: keys of the corresponding items to be transformed.
        guidance: key to the guidance. It is used to generate the bounding box of foreground
        spatial_size: minimal spatial size of the image patch e.g. [128, 128, 128] to fit in.
        margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
        meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_keys is None, use `key_{postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        start_coord_key: key to record the start coordinate of spatial bounding box for foreground.
        end_coord_key: key to record the end coordinate of spatial bounding box for foreground.
        original_shape_key: key to record original shape for foreground.
        cropped_shape_key: key to record cropped shape for foreground.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str,
        spatial_size,
        margin=20,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix=DEFAULT_POST_FIX,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        original_shape_key: str = "foreground_original_shape",
        cropped_shape_key: str = "foreground_cropped_shape",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.guidance = guidance
        self.spatial_size = list(spatial_size)
        self.margin = margin
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.original_shape_key = original_shape_key
        self.cropped_shape_key = cropped_shape_key

    def bounding_box(self, points, img_shape):
        ndim = len(img_shape)
        margin = ensure_tuple_rep(self.margin, ndim)
        for m in margin:
            if m < 0:
                raise ValueError("margin value should not be negative number.")

        box_start = [0] * ndim
        box_end = [0] * ndim

        for di in range(ndim):
            dt = points[..., di]
            min_d = max(min(dt - margin[di]), 0)
            max_d = min(img_shape[di], max(dt + margin[di] + 1))
            box_start[di], box_end[di] = min_d, max_d
        return box_start, box_end

    def __call__(self, data):
        d: Dict = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            return d

        guidance = d[self.guidance]
        original_spatial_shape = d[first_key].shape[1:]
        box_start, box_end = self.bounding_box(np.array(guidance[0] + guidance[1]), original_spatial_shape)
        center = list(np.mean([box_start, box_end], axis=0).astype(int, copy=False))
        spatial_size = self.spatial_size

        box_size = list(np.subtract(box_end, box_start).astype(int, copy=False))
        spatial_size = spatial_size[-len(box_size) :]

        if len(spatial_size) < len(box_size):
            # If the data is in 3D and spatial_size is specified as 2D [256,256]
            # Then we will get all slices in such case
            diff = len(box_size) - len(spatial_size)
            spatial_size = list(original_spatial_shape[1 : (1 + diff)]) + spatial_size

        if np.all(np.less(box_size, spatial_size)):
            if len(center) == 3:
                # 3D Deepgrow: set center to be middle of the depth dimension (D)
                center[0] = spatial_size[0] // 2
            cropper = SpatialCrop(roi_center=center, roi_size=spatial_size)
        else:
            cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)

        # update bounding box in case it was corrected by the SpatialCrop constructor
        box_start = np.array([s.start for s in cropper.slices])
        box_end = np.array([s.stop for s in cropper.slices])
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            if not np.array_equal(d[key].shape[1:], original_spatial_shape):
                raise RuntimeError("All the image specified in keys should have same spatial shape")
            meta_key = meta_key or f"{key}_{meta_key_postfix}"
            d[meta_key][self.start_coord_key] = box_start
            d[meta_key][self.end_coord_key] = box_end
            d[meta_key][self.original_shape_key] = d[key].shape

            image = cropper(d[key])
            d[meta_key][self.cropped_shape_key] = image.shape
            d[key] = image

        pos_clicks, neg_clicks = guidance[0], guidance[1]
        pos = np.subtract(pos_clicks, box_start).tolist() if len(pos_clicks) else []
        neg = np.subtract(neg_clicks, box_start).tolist() if len(neg_clicks) else []

        d[self.guidance] = [pos, neg]
        return d


class ResizeGuidanced(Transform):
    """
    Resize the guidance based on cropped vs resized image.

    This transform assumes that the images have been cropped and resized. And the shape after cropped is store inside
    the meta dict of ref image.

    Args:
        guidance: key to guidance
        ref_image: key to reference image to fetch current and original image details
        meta_keys: explicitly indicate the key of the metadata dictionary of `ref_image`.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            if None, will try to construct meta_keys by `{ref_image}_{meta_key_postfix}`.
        meta_key_postfix: if meta_key is None, use `{ref_image}_{meta_key_postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        cropped_shape_key: key that records cropped shape for foreground.
    """

    def __init__(
        self,
        guidance: str,
        ref_image: str,
        meta_keys: Optional[str] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        cropped_shape_key: str = "foreground_cropped_shape",
    ) -> None:
        self.guidance = guidance
        self.ref_image = ref_image
        self.meta_keys = meta_keys
        self.meta_key_postfix = meta_key_postfix
        self.cropped_shape_key = cropped_shape_key

    def __call__(self, data):
        d = dict(data)
        guidance = d[self.guidance]
        meta_dict: Dict = d[self.meta_keys or f"{self.ref_image}_{self.meta_key_postfix}"]
        current_shape = d[self.ref_image].shape[1:]
        cropped_shape = meta_dict[self.cropped_shape_key][1:]
        factor = np.divide(current_shape, cropped_shape)

        pos_clicks, neg_clicks = guidance[0], guidance[1]
        pos = np.multiply(pos_clicks, factor).astype(int, copy=False).tolist() if len(pos_clicks) else []
        neg = np.multiply(neg_clicks, factor).astype(int, copy=False).tolist() if len(neg_clicks) else []

        d[self.guidance] = [pos, neg]
        return d


class RestoreLabeld(MapTransform):
    """
    Restores label based on the ref image.

    The ref_image is assumed that it went through the following transforms:

        1. Fetch2DSliced (If 2D)
        2. Spacingd
        3. SpatialCropGuidanced
        4. Resized

    And its shape is assumed to be (C, D, H, W)

    This transform tries to undo these operation so that the result label can be overlapped with original volume.
    It does the following operation:

        1. Undo Resized
        2. Undo SpatialCropGuidanced
        3. Undo Spacingd
        4. Undo Fetch2DSliced

    The resulting label is of shape (D, H, W)

    Args:
        keys: keys of the corresponding items to be transformed.
        ref_image: reference image to fetch current and original image details
        slice_only: apply only to an applicable slice, in case of 2D model/prediction
        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function for padding. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            It also can be a sequence of bool, each element corresponds to a key in ``keys``.
        meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_key is None, use `key_{meta_key_postfix} to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        start_coord_key: key that records the start coordinate of spatial bounding box for foreground.
        end_coord_key: key that records the end coordinate of spatial bounding box for foreground.
        original_shape_key: key that records original shape for foreground.
        cropped_shape_key: key that records cropped shape for foreground.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        ref_image: str,
        slice_only: bool = False,
        mode: Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str] = InterpolateMode.NEAREST,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        meta_keys: Optional[str] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        original_shape_key: str = "foreground_original_shape",
        cropped_shape_key: str = "foreground_cropped_shape",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.ref_image = ref_image
        self.slice_only = slice_only
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = meta_key_postfix
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.original_shape_key = original_shape_key
        self.cropped_shape_key = cropped_shape_key

    def __call__(self, data):
        d = dict(data)
        meta_dict: Dict = d[f"{self.ref_image}_{self.meta_key_postfix}"]

        for key, mode, align_corners, meta_key in self.key_iterator(d, self.mode, self.align_corners, self.meta_keys):
            image = d[key]

            # Undo Resize
            current_shape = image.shape
            cropped_shape = meta_dict[self.cropped_shape_key]
            if np.any(np.not_equal(current_shape, cropped_shape)):
                resizer = Resize(spatial_size=cropped_shape[1:], mode=mode)
                image = resizer(image, mode=mode, align_corners=align_corners)

            # Undo Crop
            original_shape = meta_dict[self.original_shape_key]
            result = np.zeros(original_shape, dtype=np.float32)
            box_start = meta_dict[self.start_coord_key]
            box_end = meta_dict[self.end_coord_key]

            spatial_dims = min(len(box_start), len(image.shape[1:]))
            slices = [slice(None)] + [slice(s, e) for s, e in zip(box_start[:spatial_dims], box_end[:spatial_dims])]
            slices = tuple(slices)
            result[slices] = image

            # Undo Spacing
            current_size = result.shape[1:]
            # change spatial_shape from HWD to DHW
            spatial_shape = list(np.roll(meta_dict["spatial_shape"], 1))
            spatial_size = spatial_shape[-len(current_size) :]

            if np.any(np.not_equal(current_size, spatial_size)):
                resizer = Resize(spatial_size=spatial_size, mode=mode)
                result = resizer(result, mode=mode, align_corners=align_corners)

            # Undo Slicing
            slice_idx = meta_dict.get("slice_idx")
            if slice_idx is None or self.slice_only:
                final_result = result if len(result.shape) <= 3 else result[0]
            else:
                slice_idx = meta_dict["slice_idx"][0]
                final_result = np.zeros(tuple(spatial_shape))
                final_result[slice_idx] = result
            d[key] = final_result

            meta_key = meta_key or f"{key}_{self.meta_key_postfix}"
            meta = d.get(meta_key)
            if meta is None:
                meta = dict()
                d[meta_key] = meta
            meta["slice_idx"] = slice_idx
            meta["affine"] = meta_dict["original_affine"]
        return d


class Fetch2DSliced(MapTransform):
    """
    Fetch one slice in case of a 3D volume.

    The volume only contains spatial coordinates.

    Args:
        keys: keys of the corresponding items to be transformed.
        guidance: key that represents guidance.
        axis: axis that represents slice in 3D volume.
        meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: use `key_{meta_key_postfix}` to fetch the metadata according to the key data,
            default is `meta_dict`, the metadata is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys,
        guidance="guidance",
        axis: int = 0,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance = guidance
        self.axis = axis
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def _apply(self, image, guidance):
        slice_idx = guidance[2]  # (pos, neg, slice_idx)
        idx = []
        for i, size_i in enumerate(image.shape):
            idx.append(slice_idx) if i == self.axis else idx.append(slice(0, size_i))

        idx = tuple(idx)
        return image[idx], idx

    def __call__(self, data):
        d = dict(data)
        guidance = d[self.guidance]
        if len(guidance) < 3:
            raise RuntimeError("Guidance does not container slice_idx!")
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            img_slice, idx = self._apply(d[key], guidance)
            d[key] = img_slice
            d[meta_key or f"{key}_{meta_key_postfix}"]["slice_idx"] = idx
        return d
