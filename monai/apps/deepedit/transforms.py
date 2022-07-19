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
import logging
import random
import warnings
from typing import Dict, Hashable, Mapping, Optional

import numpy as np
import torch

from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.networks.layers import GaussianFilter
from monai.transforms.transform import MapTransform, Randomizable, Transform
from monai.utils import min_version, optional_import

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

logger = logging.getLogger(__name__)


distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")


class DiscardAddGuidanced(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        number_intensity_ch: int = 1,
        probability: float = 1.0,
        label_names=None,
        allow_missing_keys: bool = False,
    ):
        """
        Discard positive and negative points according to discard probability

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            number_intensity_ch: number of intensity channels
            probability: probability of discarding clicks
        """
        super().__init__(keys, allow_missing_keys)

        self.number_intensity_ch = number_intensity_ch
        self.discard_probability = probability
        self.label_names = label_names

    def _apply(self, image):
        if self.discard_probability >= 1.0 or np.random.choice(
            [True, False], p=[self.discard_probability, 1 - self.discard_probability]
        ):
            signal = np.zeros(
                (len(self.label_names), image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32
            )
            if image.shape[0] == self.number_intensity_ch + len(self.label_names):
                image[self.number_intensity_ch :, ...] = signal
            else:
                image = np.concatenate([image, signal], axis=0)
        return image

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "image":
                tmp_image = self._apply(d[key])
                if isinstance(d[key], MetaTensor):
                    d[key].array = tmp_image
                else:
                    d[key] = tmp_image
            else:
                print("This transform only applies to the image")
        return d


class NormalizeLabelsInDatasetd(MapTransform):
    def __init__(self, keys: KeysCollection, label_names=None, allow_missing_keys: bool = False):
        """
        Normalize label values according to label names dictionary

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            label_names: all label names
        """
        super().__init__(keys, allow_missing_keys)

        self.label_names = label_names

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                # Dictionary containing new label numbers
                new_label_names = {}
                label = np.zeros(d[key].shape)
                # Making sure the range values and number of labels are the same
                for idx, (key_label, val_label) in enumerate(self.label_names.items(), start=1):
                    if key_label != "background":
                        new_label_names[key_label] = idx
                        label[d[key] == val_label] = idx
                    if key_label == "background":
                        new_label_names["background"] = 0

                d["label_names"] = new_label_names
                d[key] = label
            else:
                warnings.warn("This transform only applies to the label")
        return d


class SingleLabelSelectiond(MapTransform):
    def __init__(self, keys: KeysCollection, label_names=None, allow_missing_keys: bool = False):
        """
        Selects one label at a time to train the DeepEdit

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            label_names: all label names
        """
        super().__init__(keys, allow_missing_keys)

        self.label_names = label_names
        self.all_label_values = {
            "spleen": 1,
            "right kidney": 2,
            "left kidney": 3,
            "gallbladder": 4,
            "esophagus": 5,
            "liver": 6,
            "stomach": 7,
            "aorta": 8,
            "inferior vena cava": 9,
            "portal_vein": 10,
            "splenic_vein": 11,
            "pancreas": 12,
            "right adrenal gland": 13,
            "left adrenal gland": 14,
        }

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                # Taking one label at a time
                t_label = np.random.choice(self.label_names)
                d["current_label"] = t_label
                d[key][d[key] != self.all_label_values[t_label]] = 0.0
                # Convert label to index values following label_names argument
                max_label_val = self.label_names.index(t_label) + 1
                d[key][d[key] > 0] = max_label_val
                print(f"Using label {t_label} with number: {d[key].max()}")
            else:
                warnings.warn("This transform only applies to the label")
        return d


class AddGuidanceSignalDeepEditd(MapTransform):
    """
    Add Guidance signal for input image. Multilabel DeepEdit

    Based on the "guidance" points, apply Gaussian to them and add them as new channel for input image.

    Args:
        guidance: key to store guidance.
        sigma: standard deviation for Gaussian kernel.
        number_intensity_ch: channel index.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sigma: int = 3,
        number_intensity_ch: int = 1,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance = guidance
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch

    def _get_signal(self, image, guidance):
        dimensions = 3 if len(image.shape) > 3 else 2
        guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        guidance = json.loads(guidance) if isinstance(guidance, str) else guidance

        # In inference the user may not provide clicks for some channels/labels
        if len(guidance):
            if dimensions == 3:
                # Assume channel is first and depth is last CHWD
                signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
            else:
                signal = np.zeros((1, image.shape[-2], image.shape[-1]), dtype=np.float32)

            sshape = signal.shape
            for point in guidance:  # TO DO: make the guidance a list only - it is currently a list of list
                if np.any(np.asarray(point) < 0):
                    continue

                if dimensions == 3:
                    # Making sure points fall inside the image dimension
                    p1 = max(0, min(int(point[-3]), sshape[-3] - 1))
                    p2 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p3 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2, p3] = 1.0
                else:
                    p1 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p2 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2] = 1.0

            # Apply a Gaussian filter to the signal
            if np.max(signal[0]) > 0:
                signal_tensor = torch.tensor(signal[0])
                pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                signal_tensor = signal_tensor.squeeze(0).squeeze(0)
                signal[0] = signal_tensor.detach().cpu().numpy()
                signal[0] = (signal[0] - np.min(signal[0])) / (np.max(signal[0]) - np.min(signal[0]))
            return signal
        else:
            if dimensions == 3:
                signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
            else:
                signal = np.zeros((1, image.shape[-2], image.shape[-1]), dtype=np.float32)
            return signal

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "image":
                image = d[key]
                tmp_image = image[0 : 0 + self.number_intensity_ch, ...]
                guidance = d[self.guidance]
                for key_label in guidance.keys():
                    # Getting signal based on guidance
                    signal = self._get_signal(image, guidance[key_label])
                    tmp_image = np.concatenate([tmp_image, signal], axis=0)
                    if isinstance(d[key], MetaTensor):
                        d[key].array = tmp_image
                    else:
                        d[key] = tmp_image
                return d
            else:
                print("This transform only applies to image key")
        return d


class FindAllValidSlicesDeepEditd(MapTransform):
    """
    Find/List all valid slices in the labels.
    Label is assumed to be a 4D Volume with shape CHWD, where C=1.

    Args:
        sids: key to store slices indices having valid label map.
    """

    def __init__(self, keys: KeysCollection, sids="sids", allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.sids = sids

    def _apply(self, label, d):
        sids = {}
        for key_label in d["label_names"].keys():
            l_ids = []
            for sid in range(label.shape[-1]):  # Assume channel is first and depth is last CHWD
                if d["label_names"][key_label] in label[0][..., sid]:
                    l_ids.append(sid)
            sids[key_label] = l_ids
        return sids

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label = d[key]
                if label.shape[0] != 1:
                    raise ValueError("Only supports single channel labels!")

                if len(label.shape) != 4:  # only for 3D
                    raise ValueError("Only supports label with shape CHWD!")

                sids = self._apply(label, d)
                if sids is not None and len(sids.keys()):
                    d[self.sids] = sids
                return d
            else:
                print("This transform only applies to label key")
        return d


class AddInitialSeedPointDeepEditd(Randomizable, MapTransform):
    """
    Add random guidance as initial seed point for a given label.

    Note that the label is of size (C, D, H, W) or (C, H, W)

    The guidance is of size (2, N, # of dims) where N is number of guidance added.
    # of dims = 4 when C, D, H, W; # of dims = 3 when (C, H, W)

    Args:
        guidance: key to store guidance.
        sids: key that represents lists of valid slice indices for the given label.
        sid: key that represents the slice to add initial seed point.  If not present, random sid will be chosen.
        connected_regions: maximum connected regions to use for adding initial points.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sids: str = "sids",
        sid: str = "sid",
        connected_regions: int = 5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sids_key = sids
        self.sid_key = sid
        self.sid: Dict[str, int] = dict()
        self.guidance = guidance
        self.connected_regions = connected_regions

    def _apply(self, label, sid, key_label):
        dimensions = 3 if len(label.shape) > 3 else 2
        self.default_guidance = [-1] * (dimensions + 1)

        dims = dimensions
        if sid is not None and dimensions == 3:
            dims = 2
            label = label[0][..., sid][np.newaxis]  # Assume channel is first and depth is last CHWD

        # THERE MAY BE MULTIPLE BLOBS FOR SINGLE LABEL IN THE SELECTED SLICE
        label = (label > 0.5).astype(np.float32)
        # measure.label: Label connected regions of an integer array - Two pixels are connected
        # when they are neighbors and have the same value
        blobs_labels = measure.label(label.astype(int), background=0) if dims == 2 else label
        if np.max(blobs_labels) <= 0:
            raise AssertionError(f"SLICES NOT FOUND FOR LABEL: {key_label}")

        pos_guidance = []
        for ridx in range(1, 2 if dims == 3 else self.connected_regions + 1):
            if dims == 2:
                label = (blobs_labels == ridx).astype(np.float32)
                if np.sum(label) == 0:
                    pos_guidance.append(self.default_guidance)
                    continue

            # The distance transform provides a metric or measure of the separation of points in the image.
            # This function calculates the distance between each pixel that is set to off (0) and
            # the nearest nonzero pixel for binary images - http://matlab.izmiran.ru/help/toolbox/images/morph14.html
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
                # Clicks are created using this convention Channel Height Width Depth (CHWD)
                pos_guidance.append([g[0], g[-2], g[-1], sid])  # Assume channel is first and depth is last CHWD

        return np.asarray([pos_guidance])

    def _randomize(self, d, key_label):
        sids = d.get(self.sids_key).get(key_label) if d.get(self.sids_key) is not None else None
        sid = d.get(self.sid_key).get(key_label) if d.get(self.sid_key) is not None else None
        if sids is not None and sids:
            if sid is None or sid not in sids:
                sid = self.R.choice(sids, replace=False)
        else:
            logger.info(f"Not slice IDs for label: {key_label}")
            sid = None
        self.sid[key_label] = sid

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label_guidances = {}
                for key_label in d["sids"].keys():
                    # Randomize: Select a random slice
                    self._randomize(d, key_label)
                    # Generate guidance base on selected slice
                    tmp_label = np.copy(d[key])
                    # Taking one label to create the guidance
                    if key_label != "background":
                        tmp_label[tmp_label != float(d["label_names"][key_label])] = 0
                    else:
                        tmp_label[tmp_label != float(d["label_names"][key_label])] = 1
                        tmp_label = 1 - tmp_label
                    label_guidances[key_label] = json.dumps(
                        self._apply(tmp_label, self.sid.get(key_label), key_label).astype(int).tolist()
                    )
                d[self.guidance] = label_guidances
                return d
            else:
                print("This transform only applies to label key")
        return d


class FindDiscrepancyRegionsDeepEditd(MapTransform):
    """
    Find discrepancy between prediction and actual during click interactions during training.

    Args:
        pred: key to prediction source.
        discrepancy: key to store discrepancies found between label and prediction.
    """

    def __init__(
        self,
        keys: KeysCollection,
        pred: str = "pred",
        discrepancy: str = "discrepancy",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.pred = pred
        self.discrepancy = discrepancy

    @staticmethod
    def disparity(label, pred):
        disparity = label - pred
        # Negative ONES mean predicted label is not part of the ground truth
        # Positive ONES mean predicted label missed that region of the ground truth
        pos_disparity = (disparity > 0).astype(np.float32)
        neg_disparity = (disparity < 0).astype(np.float32)
        return [pos_disparity, neg_disparity]

    def _apply(self, label, pred):
        return self.disparity(label, pred)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                all_discrepancies = {}
                for _, (key_label, val_label) in enumerate(d["label_names"].items()):
                    if key_label != "background":
                        # Taking single label
                        label = np.copy(d[key])
                        label[label != val_label] = 0
                        # Label should be represented in 1
                        label = (label > 0.5).astype(np.float32)
                        # Taking single prediction
                        pred = np.copy(d[self.pred])
                        pred[pred != val_label] = 0
                        # Prediction should be represented in one
                        pred = (pred > 0.5).astype(np.float32)
                    else:
                        # Taking single label
                        label = np.copy(d[key])
                        label[label != val_label] = 1
                        label = 1 - label
                        # Label should be represented in 1
                        label = (label > 0.5).astype(np.float32)
                        # Taking single prediction
                        pred = np.copy(d[self.pred])
                        pred[pred != val_label] = 1
                        pred = 1 - pred
                        # Prediction should be represented in one
                        pred = (pred > 0.5).astype(np.float32)
                    all_discrepancies[key_label] = self._apply(label, pred)
                d[self.discrepancy] = all_discrepancies
                return d
            else:
                print("This transform only applies to 'label' key")
        return d


class AddRandomGuidanceDeepEditd(Randomizable, MapTransform):
    """
    Add random guidance based on discrepancies that were found between label and prediction.

    Args:
        guidance: key to guidance source, shape (2, N, # of dim)
        discrepancy: key to discrepancy map between label and prediction shape (2, C, H, W, D) or (2, C, H, W)
        probability: key to click/interaction probability, shape (1)
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        discrepancy: str = "discrepancy",
        probability: str = "probability",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance = guidance
        self.discrepancy = discrepancy
        self.probability = probability
        self._will_interact = None
        self.is_pos = None
        self.is_other = None
        self.default_guidance = None

    def randomize(self, data=None):
        probability = data[self.probability]
        self._will_interact = self.R.choice([True, False], p=[probability, 1.0 - probability])

    def find_guidance(self, discrepancy):
        distance = distance_transform_cdt(discrepancy).flatten()
        probability = np.exp(distance.flatten()) - 1.0
        idx = np.where(discrepancy.flatten() > 0)[0]

        if np.sum(discrepancy > 0) > 0:
            seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
            dst = distance[seed]

            g = np.asarray(np.unravel_index(seed, discrepancy.shape)).transpose().tolist()[0]
            g[0] = dst[0]
            return g
        return None

    def add_guidance(self, guidance, discrepancy, label_names, labels):

        # Positive clicks of the segment in the iteration
        pos_discr = discrepancy[0]  # idx 0 is positive discrepancy and idx 1 is negative discrepancy

        # Check the areas that belong to other segments
        other_discrepancy_areas = {}
        for _, (key_label, val_label) in enumerate(label_names.items()):
            if key_label != "background":
                tmp_label = np.copy(labels)
                tmp_label[tmp_label != val_label] = 0
                tmp_label = (tmp_label > 0.5).astype(np.float32)
                other_discrepancy_areas[key_label] = np.sum(discrepancy[1] * tmp_label)
            else:
                tmp_label = np.copy(labels)
                tmp_label[tmp_label != val_label] = 1
                tmp_label = 1 - tmp_label
                other_discrepancy_areas[key_label] = np.sum(discrepancy[1] * tmp_label)

        # Add guidance to the current key label
        if np.sum(pos_discr) > 0:
            guidance.append(self.find_guidance(pos_discr))
            self.is_pos = True

        # Add guidance to the other areas
        for key_label in label_names.keys():
            # Areas that cover more than 50 voxels
            if other_discrepancy_areas[key_label] > 50:
                self.is_other = True
                if key_label != "background":
                    tmp_label = np.copy(labels)
                    tmp_label[tmp_label != label_names[key_label]] = 0
                    tmp_label = (tmp_label > 0.5).astype(np.float32)
                    self.tmp_guidance[key_label].append(self.find_guidance(discrepancy[1] * tmp_label))
                else:
                    tmp_label = np.copy(labels)
                    tmp_label[tmp_label != label_names[key_label]] = 1
                    tmp_label = 1 - tmp_label
                    self.tmp_guidance[key_label].append(self.find_guidance(discrepancy[1] * tmp_label))

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        guidance = d[self.guidance]
        discrepancy = d[self.discrepancy]
        self.randomize(data)
        if self._will_interact:
            # Convert all guidance to lists so new guidance can be easily appended
            self.tmp_guidance = {}
            for key_label in d["label_names"].keys():
                tmp_gui = guidance[key_label]
                tmp_gui = tmp_gui.tolist() if isinstance(tmp_gui, np.ndarray) else tmp_gui
                tmp_gui = json.loads(tmp_gui) if isinstance(tmp_gui, str) else tmp_gui
                self.tmp_guidance[key_label] = [j for j in tmp_gui if -1 not in j]

            # Add guidance according to discrepancy
            for key_label in d["label_names"].keys():
                # Add guidance based on discrepancy
                self.add_guidance(self.tmp_guidance[key_label], discrepancy[key_label], d["label_names"], d["label"])

            # Checking the number of clicks
            num_clicks = random.randint(1, 10)
            counter = 0
            keep_guidance = []
            while True:
                aux_label = random.choice(list(d["label_names"].keys()))
                if aux_label in keep_guidance:
                    pass
                else:
                    keep_guidance.append(aux_label)
                    counter = counter + len(self.tmp_guidance[aux_label])
                    # If collected clicks is bigger than max clicks, discard the others
                    if counter >= num_clicks:
                        for key_label in d["label_names"].keys():
                            if key_label not in keep_guidance:
                                self.tmp_guidance[key_label] = []
                        logger.info(f"Number of simulated clicks: {counter}")
                        break

                # Breaking once all labels are covered
                if len(keep_guidance) == len(d["label_names"].keys()):
                    logger.info(f"Number of simulated clicks: {counter}")
                    break

        return d


class AddGuidanceFromPointsDeepEditd(Transform):
    """
    Add guidance based on user clicks. ONLY WORKS FOR 3D

    We assume the input is loaded by LoadImaged and has the shape of (H, W, D) originally.
    Clicks always specify the coordinates in (H, W, D)

    Args:
        ref_image: key to reference image to fetch current and original image details.
        guidance: output key to store guidance.
        meta_keys: explicitly indicate the key of the metadata dictionary of `ref_image`.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            if None, will try to construct meta_keys by `{ref_image}_{meta_key_postfix}`.
        meta_key_postfix: if meta_key is None, use `{ref_image}_{meta_key_postfix}` to to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.

    """

    def __init__(
        self,
        ref_image,
        guidance: str = "guidance",
        label_names=None,
        meta_keys: Optional[str] = None,
        meta_key_postfix: str = "meta_dict",
    ):
        self.ref_image = ref_image
        self.guidance = guidance
        self.label_names = label_names
        self.meta_keys = meta_keys
        self.meta_key_postfix = meta_key_postfix

    @staticmethod
    def _apply(clicks, factor):
        if len(clicks):
            guidance = np.multiply(clicks, factor).astype(int).tolist()
            return guidance
        else:
            return []

    def __call__(self, data):
        d = dict(data)
        meta_dict_key = self.meta_keys or f"{self.ref_image}_{self.meta_key_postfix}"
        if meta_dict_key not in d:
            raise RuntimeError(f"Missing meta_dict {meta_dict_key} in data!")
        if "spatial_shape" not in d[meta_dict_key]:
            raise RuntimeError('Missing "spatial_shape" in meta_dict!')

        # Assume channel is first and depth is last CHWD
        original_shape = d[meta_dict_key]["spatial_shape"]
        current_shape = list(d[self.ref_image].shape)[1:]

        # in here we assume the depth dimension is in the last dimension of "original_shape" and "current_shape"
        factor = np.array(current_shape) / original_shape

        # Creating guidance for all clicks
        all_guidances = {}
        for key_label in self.label_names.keys():
            clicks = d.get(key_label, [])
            clicks = list(np.array(clicks).astype(int))
            all_guidances[key_label] = self._apply(clicks, factor)
        d[self.guidance] = all_guidances
        return d


class ResizeGuidanceMultipleLabelDeepEditd(Transform):
    """
    Resize the guidance based on cropped vs resized image.

    """

    def __init__(self, guidance: str, ref_image: str) -> None:
        self.guidance = guidance
        self.ref_image = ref_image

    def __call__(self, data):
        d = dict(data)
        # Assume channel is first and depth is last CHWD
        current_shape = d[self.ref_image].shape[1:]
        original_shape = d["image_meta_dict"]["spatial_shape"]

        factor = np.divide(current_shape, original_shape)
        all_guidances = {}
        for key_label in d[self.guidance].keys():
            guidance = (
                np.multiply(d[self.guidance][key_label], factor).astype(int).tolist()
                if len(d[self.guidance][key_label])
                else []
            )
            all_guidances[key_label] = guidance

        d[self.guidance] = all_guidances
        return d


class SplitPredsLabeld(MapTransform):
    """
    Split preds and labels for individual evaluation

    """

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "pred":
                for idx, (key_label, _) in enumerate(d["label_names"].items()):
                    if key_label != "background":
                        d[f"pred_{key_label}"] = d[key][idx + 1, ...][None]
                        d[f"label_{key_label}"] = d["label"][idx + 1, ...][None]
            elif key != "pred":
                logger.info("This is only for pred key")
        return d


class AddInitialSeedPointMissingLabelsd(Randomizable, MapTransform):
    """
    Add random guidance as initial seed point for a given label.
    Note that the label is of size (C, D, H, W) or (C, H, W)
    The guidance is of size (2, N, # of dims) where N is number of guidance added.
    # of dims = 4 when C, D, H, W; # of dims = 3 when (C, H, W)
    Args:
        guidance: key to store guidance.
        sids: key that represents lists of valid slice indices for the given label.
        sid: key that represents the slice to add initial seed point.  If not present, random sid will be chosen.
        connected_regions: maximum connected regions to use for adding initial points.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sids: str = "sids",
        sid: str = "sid",
        connected_regions: int = 5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sids_key = sids
        self.sid_key = sid
        self.sid: Dict[str, int] = dict()
        self.guidance = guidance
        self.connected_regions = connected_regions

    def _apply(self, label, sid):
        dimensions = 3 if len(label.shape) > 3 else 2
        self.default_guidance = [-1] * (dimensions + 1)

        dims = dimensions
        if sid is not None and dimensions == 3:
            dims = 2
            label = label[0][..., sid][np.newaxis]  # Assume channel is first and depth is last CHWD

        # THERE MAY BE MULTIPLE BLOBS FOR SINGLE LABEL IN THE SELECTED SLICE
        label = (label > 0.5).astype(np.float32)
        # measure.label: Label connected regions of an integer array - Two pixels are connected
        # when they are neighbors and have the same value
        blobs_labels = measure.label(label.astype(int), background=0) if dims == 2 else label

        label_guidance = []
        # If there are is presence of that label in this slice
        if np.max(blobs_labels) <= 0:
            label_guidance.append(self.default_guidance)
        else:
            for ridx in range(1, 2 if dims == 3 else self.connected_regions + 1):
                if dims == 2:
                    label = (blobs_labels == ridx).astype(np.float32)
                    if np.sum(label) == 0:
                        label_guidance.append(self.default_guidance)
                        continue

                # The distance transform provides a metric or measure of the separation of points in the image.
                # This function calculates the distance between each pixel that is set to off (0) and
                # the nearest nonzero pixel for binary images
                # http://matlab.izmiran.ru/help/toolbox/images/morph14.html
                distance = distance_transform_cdt(label).flatten()
                probability = np.exp(distance) - 1.0

                idx = np.where(label.flatten() > 0)[0]
                seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
                dst = distance[seed]

                g = np.asarray(np.unravel_index(seed, label.shape)).transpose().tolist()[0]
                g[0] = dst[0]  # for debug
                if dimensions == 2 or dims == 3:
                    label_guidance.append(g)
                else:
                    # Clicks are created using this convention Channel Height Width Depth (CHWD)
                    label_guidance.append([g[0], g[-2], g[-1], sid])  # Assume channel is first and depth is last CHWD

        return np.asarray(label_guidance)

    def _randomize(self, d, key_label):
        sids = d.get(self.sids_key).get(key_label) if d.get(self.sids_key) is not None else None
        sid = d.get(self.sid_key).get(key_label) if d.get(self.sid_key) is not None else None
        if sids is not None and sids:
            if sid is None or sid not in sids:
                sid = self.R.choice(sids, replace=False)
        else:
            logger.info(f"Not slice IDs for label: {key_label}")
            sid = None
        self.sid[key_label] = sid

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label_guidances = {}
                for key_label in d["sids"].keys():
                    # Randomize: Select a random slice
                    self._randomize(d, key_label)
                    # Generate guidance base on selected slice
                    tmp_label = np.copy(d[key])
                    # Taking one label to create the guidance
                    if key_label != "background":
                        tmp_label[tmp_label != float(d["label_names"][key_label])] = 0
                    else:
                        tmp_label[tmp_label != float(d["label_names"][key_label])] = 1
                        tmp_label = 1 - tmp_label
                    label_guidances[key_label] = json.dumps(
                        self._apply(tmp_label, self.sid.get(key_label)).astype(int).tolist()
                    )
                d[self.guidance] = label_guidances
                return d
            else:
                print("This transform only applies to label key")
        return d


class FindAllValidSlicesMissingLabelsd(MapTransform):
    """
    Find/List all valid slices in the labels.
    Label is assumed to be a 4D Volume with shape CHWD, where C=1.
    Args:
        sids: key to store slices indices having valid label map.
    """

    def __init__(self, keys: KeysCollection, sids="sids", allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.sids = sids

    def _apply(self, label, d):
        sids = {}
        for key_label in d["label_names"].keys():
            l_ids = []
            for sid in range(label.shape[-1]):  # Assume channel is first and depth is last CHWD
                if d["label_names"][key_label] in label[0][..., sid]:
                    l_ids.append(sid)
            # If there are not slices with the label
            if l_ids == []:
                l_ids = [-1] * 10
            sids[key_label] = l_ids
        return sids

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label = d[key]
                if label.shape[0] != 1:
                    raise ValueError("Only supports single channel labels!")

                if len(label.shape) != 4:  # only for 3D
                    raise ValueError("Only supports label with shape CHWD!")

                sids = self._apply(label, d)
                if sids is not None and len(sids.keys()):
                    d[self.sids] = sids
                return d
            else:
                print("This transform only applies to label key")
        return d
