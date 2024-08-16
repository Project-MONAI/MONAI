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

from typing import Sequence

import numpy as np

from monai.config import DtypeLike, KeysCollection
from monai.transforms import MapLabelValue
from monai.transforms.transform import MapTransform
from monai.utils import look_up_option


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


class VistaPreTransform(MapTransform):
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

        Args:
            keys: keys of the corresponding items to be transformed.
            dataset_transforms: a dictionary specifies the transform for corresponding dataset:
                key: dataset name, value: list of data transforms.
            dataset_key: key to get the dataset name from the data dictionary, default to "dataset_name".
            allow_missing_keys: don't raise exception if key is missing.
            special_index: the class index that need to be handled differently.
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
            pass

        return data


class RelabelD(MapTransform):
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
                global class indices, format:
                key: dataset name.
                value: list of (local label, global label) pairs. This list of local -> global label mappings
                    will be applied to each input `data[keys]`. If `data[dataset_key]` is not in `label_mappings`,
                    label_mappings['default']` will be used. if `label_mappings[data[dataset_key]]` is None,
                    no relabeling will be performed.
                set `label_mappings={}` to completely skip this transform.
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