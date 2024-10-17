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

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection
from monai.transforms import MapLabelValue
from monai.transforms.transform import MapTransform
from monai.transforms.utils import keep_components_with_positive_points
from monai.utils import look_up_option

__all__ = ["VistaPreTransformd", "VistaPostTransformd", "Relabeld"]


def _get_name_to_index_mapping(labels_dict: dict | None) -> dict:
    """get the label name to index mapping"""
    name_to_index_mapping = {}
    if labels_dict is not None:
        name_to_index_mapping = {v.lower(): int(k) for k, v in labels_dict.items()}
    return name_to_index_mapping


def _convert_name_to_index(name_to_index_mapping: dict, label_prompt: list | None) -> list | None:
    """convert the label name to index"""
    if label_prompt is not None and isinstance(label_prompt, list):
        converted_label_prompt = []
        # for new class, add to the mapping
        for l in label_prompt:
            if isinstance(l, str) and not l.isdigit():
                if l.lower() not in name_to_index_mapping:
                    name_to_index_mapping[l.lower()] = len(name_to_index_mapping)
        for l in label_prompt:
            if isinstance(l, (int, str)):
                converted_label_prompt.append(
                    name_to_index_mapping.get(l.lower(), int(l) if l.isdigit() else 0) if isinstance(l, str) else int(l)
                )
            else:
                converted_label_prompt.append(l)
        return converted_label_prompt
    return label_prompt


class VistaPreTransformd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        special_index: Sequence[int] = (25, 26, 27, 28, 29, 117),
        labels_dict: dict | None = None,
        subclass: dict | None = None,
    ) -> None:
        """
        Pre-transform for Vista3d.

        It performs two functionalities:

        1. If label prompt shows the points belong to special class (defined by special index, e.g. tumors, vessels),
           convert point labels from 0 (negative), 1 (positive) to special 2 (negative), 3 (positive).

        2. If label prompt is within the keys in subclass, convert the label prompt to its subclasses defined by subclass[key].
           e.g. "lung" label is converted to ["left lung", "right lung"].

        The `label_prompt` is a list of int values of length [B] and `point_labels` is a list of length B,
        where each element is an int value of length [B, N].

        Args:
            keys: keys of the corresponding items to be transformed.
            special_index: the index that defines the special class.
            subclass: a dictionary that maps a label prompt to its subclasses.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.special_index = special_index
        self.subclass = subclass
        self.name_to_index_mapping = _get_name_to_index_mapping(labels_dict)

    def __call__(self, data):
        label_prompt = data.get("label_prompt", None)
        point_labels = data.get("point_labels", None)
        # convert the label name to index if needed
        label_prompt = _convert_name_to_index(self.name_to_index_mapping, label_prompt)
        try:
            # The evaluator will check prompt. The invalid prompt will be skipped here and captured by evaluator.
            if self.subclass is not None and label_prompt is not None:
                _label_prompt = []
                subclass_keys = list(map(int, self.subclass.keys()))
                for i in range(len(label_prompt)):
                    if label_prompt[i] in subclass_keys:
                        _label_prompt.extend(self.subclass[str(label_prompt[i])])
                    else:
                        _label_prompt.append(label_prompt[i])
                data["label_prompt"] = _label_prompt
            if label_prompt is not None and point_labels is not None:
                if label_prompt[0] in self.special_index:
                    point_labels = np.array(point_labels)
                    point_labels[point_labels == 0] = 2
                    point_labels[point_labels == 1] = 3
                    point_labels = point_labels.tolist()
                data["point_labels"] = point_labels
        except Exception:
            # There is specific requirements for `label_prompt` and `point_labels`.
            # If B > 1 or `label_prompt` is in subclass_keys, `point_labels` must be None.
            # Those formatting errors should be captured later.
            warnings.warn("VistaPreTransformd failed to transform label prompt or point labels.")

        return data


class VistaPostTransformd(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Post-transform for Vista3d. It converts the model output logits into final segmentation masks.
        If `label_prompt` is None, the output will be thresholded to be sequential indexes [0,1,2,...],
        else the indexes will be [0, label_prompt[0], label_prompt[1], ...].
        If `label_prompt` is None while `points` are provided, the model will perform postprocess to remove
        regions that does not contain positive points.

        Args:
            keys: keys of the corresponding items to be transformed.
            dataset_transforms: a dictionary specifies the transform for corresponding dataset:
                key: dataset name, value: list of data transforms.
            dataset_key: key to get the dataset name from the data dictionary, default to "dataset_name".
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        """data["label_prompt"] should not contain 0"""
        for keys in self.keys:
            if keys in data:
                pred = data[keys]
                object_num = pred.shape[0]
                device = pred.device
                if data.get("label_prompt", None) is None and data.get("points", None) is not None:
                    pred = keep_components_with_positive_points(
                        pred.unsqueeze(0),
                        point_coords=data.get("points").to(device),
                        point_labels=data.get("point_labels").to(device),
                    )[0]
                pred[pred < 0] = 0.0
                # if it's multichannel, perform argmax
                if object_num > 1:
                    # concate background channel. Make sure user did not provide 0 as prompt.
                    is_bk = torch.all(pred <= 0, dim=0, keepdim=True)
                    pred = pred.argmax(0).unsqueeze(0).float() + 1.0
                    pred[is_bk] = 0.0
                else:
                    # AsDiscrete will remove NaN
                    # pred = monai.transforms.AsDiscrete(threshold=0.5)(pred)
                    pred[pred > 0] = 1.0
                if "label_prompt" in data and data["label_prompt"] is not None:
                    pred += 0.5  # inplace mapping to avoid cloning pred
                    label_prompt = data["label_prompt"].to(device)  # Ensure label_prompt is on the same device
                    for i in range(1, object_num + 1):
                        frac = i + 0.5
                        pred[pred == frac] = label_prompt[i - 1].to(pred.dtype)
                    pred[pred == 0.5] = 0.0
                data[keys] = pred
        return data


class Relabeld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        label_mappings: dict[str, list[tuple[int, int]]],
        dtype: DtypeLike = np.int16,
        dataset_key: str = "dataset_name",
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Remap the voxel labels in the input data dictionary based on the specified mapping.

        This list of local -> global label mappings will be applied to each input `data[keys]`.
        if `data[dataset_key]` is not in `label_mappings`, label_mappings['default']` will be used.
        if `label_mappings[data[dataset_key]]` is None, no relabeling will be performed.

        Args:
            keys: keys of the corresponding items to be transformed.
            label_mappings: a dictionary specifies how local dataset class indices are mapped to the
                global class indices. The dictionary keys are dataset names and the values are lists of
                list of (local label, global label) pairs. This list of local -> global label mappings
                will be applied to each input `data[keys]`. If `data[dataset_key]` is not in `label_mappings`,
                label_mappings['default']` will be used. if `label_mappings[data[dataset_key]]` is None,
                no relabeling will be performed. Please set `label_mappings={}` to completely skip this transform.
            dtype: convert the output data to dtype, default to float32.
            dataset_key: key to get the dataset name from the data dictionary, default to "dataset_name".
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.mappers = {}
        self.dataset_key = dataset_key
        for name, mapping in label_mappings.items():
            self.mappers[name] = MapLabelValue(
                orig_labels=[int(pair[0]) for pair in mapping],
                target_labels=[int(pair[1]) for pair in mapping],
                dtype=dtype,
            )

    def __call__(self, data):
        d = dict(data)
        dataset_name = d.get(self.dataset_key, "default")
        _m = look_up_option(dataset_name, self.mappers, default=None)
        if _m is None:
            return d
        for key in self.key_iterator(d):
            d[key] = _m(d[key])
        return d
