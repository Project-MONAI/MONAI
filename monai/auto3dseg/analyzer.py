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

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import torch

from monai.auto3dseg.operations import Operations, SampleOperations, SummaryOperations
from monai.auto3dseg.utils import (
    concat_multikeys_to_dict,
    concat_val_to_np,
    get_foreground_image,
    get_foreground_label,
    get_label_ccp,
)
from monai.bundle.config_parser import ConfigParser
from monai.bundle.utils import ID_SEP_KEY
from monai.transforms import transform
from monai.utils.enums import IMAGE_STATS, LABEL_STATS
from monai.utils.misc import label_union


class Analyzer(transform.MapTransform, ABC):
    def __init__(self, report_format):
        self.report_format = report_format
        self.ops = ConfigParser({})

    def update_ops(self, key: str, op):
        """
        Register an statistical operation to the Analyzer and update the report_format

        Args:
            key: value key in the report
            op: Operation object

        """
        self.ops[key] = op
        parser = ConfigParser(self.report_format)

        if parser.get(key, "NA") != "NA":
            parser[key] = op

        self.report_format = parser.config

    def update_ops_nested_label(self, nested_key, op):
        """
        Update operations for nested label format. Operation value in report_format will be resolved to a dict with only keys

        Args:
            nested_key: str that has format of 'key1#0#key2'
            op: statistical operation
        """
        keys = nested_key.split(ID_SEP_KEY)
        if len(keys) != 3:
            raise ValueError("Nested_key input format is wrong. Please ensure it is like key1#0#key2")

        (root, _, child_key) = keys
        if root not in self.ops:
            self.ops[root] = [{}]
        self.ops[root][0].update({child_key: None})

        self.ops[nested_key] = op

        parser = ConfigParser(self.report_format)
        if parser.get(nested_key, "NA") != "NA":
            parser[nested_key] = op

    def get_report_format(self):
        """
        Get the report format by resolving the registered operations.

        Returns:
            a dictionary with keys-None pairs

        """
        self.resolve_ops(self.report_format)
        return self.report_format

    @staticmethod
    def unwrap_ops(func):
        ret = dict.fromkeys([key for key in func.data])
        if hasattr(func, "data_addon"):
            for key in func.data_addon:
                ret.update({key: None})
        return ret

    def resolve_ops(self, report: dict):
        for k, v in report.items():
            if issubclass(v.__class__, Operations):
                report[k] = self.unwrap_ops(v)
            elif isinstance(v, list) and len(v) > 0:
                self.resolve_ops(v[0])
            else:
                report[k] = v

    @abstractmethod
    def __call__(self, data):
        """Analyze the dict format dataset, return the summary report"""
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class ImageStatsCaseAnalyzer(Analyzer):
    def __init__(self, image_key, meta_key_postfix="_meta_dict"):

        self.image_key = image_key
        self.image_meta_key = self.image_key + meta_key_postfix

        report_format = {
            IMAGE_STATS.SHAPE: None,
            IMAGE_STATS.CHANNELS: None,
            IMAGE_STATS.CROPPED_SHAPE: None,
            IMAGE_STATS.SPACING: None,
            IMAGE_STATS.INTENSITY: None,
        }

        super().__init__(report_format)
        self.update_ops(IMAGE_STATS.INTENSITY, SampleOperations())

    def __call__(self, data):
        # from time import time
        # start = time.time()
        ndas = data[self.image_key]
        ndas = [ndas[i] for i in range(ndas.shape[0])]
        if "nda_croppeds" not in data:
            data["nda_croppeds"] = [get_foreground_image(nda) for nda in ndas]
        nda_croppeds = data["nda_croppeds"]

        # perform calculation
        analysis = deepcopy(self.get_report_format())

        analysis[IMAGE_STATS.SHAPE] = [list(nda.shape) for nda in ndas]
        analysis[IMAGE_STATS.CHANNELS] = len(ndas)
        analysis[IMAGE_STATS.CROPPED_SHAPE] = [list(nda_c.shape) for nda_c in nda_croppeds]
        analysis[IMAGE_STATS.SPACING] = np.tile(
            np.diag(data[self.image_meta_key]["affine"])[:3], [len(ndas), 1]
        ).tolist()
        analysis[IMAGE_STATS.INTENSITY] = [self.ops[IMAGE_STATS.INTENSITY].evaluate(nda_c) for nda_c in nda_croppeds]

        # logger.debug(f"Get image stats spent {time.time()-start}")
        return analysis


class FgImageStatsCasesAnalyzer(Analyzer):
    def __init__(self, image_key, label_key, meta_key_postfix="_meta_dict"):

        self.image_key = image_key
        self.label_key = label_key
        self.image_meta_key = self.image_key + meta_key_postfix
        self.label_meta_key = self.label_key + meta_key_postfix

        report_format = {IMAGE_STATS.INTENSITY: None}

        super().__init__(report_format)
        self.update_ops(IMAGE_STATS.INTENSITY, SampleOperations())

    def __call__(self, data):

        ndas = data[self.image_key]  # (1,H,W,D) or (C,H,W,D)
        ndas = [ndas[i] for i in range(ndas.shape[0])]
        ndas_label = data[self.label_key]  # (H,W,D)
        nda_foregrounds = [get_foreground_label(nda, ndas_label) for nda in ndas]

        # perform calculation
        analysis = deepcopy(self.get_report_format())

        analysis[IMAGE_STATS.INTENSITY] = [self.ops[IMAGE_STATS.INTENSITY].evaluate(nda_f) for nda_f in nda_foregrounds]
        return analysis


class LabelStatsCaseAnalyzer(Analyzer):
    def __init__(self, image_key, label_key, meta_key_postfix="_meta_dict", do_ccp: bool = True):

        self.image_key = image_key
        self.label_key = label_key
        self.image_meta_key = self.image_key + meta_key_postfix
        self.label_meta_key = self.label_key + meta_key_postfix
        self.do_ccp = do_ccp

        report_format = {
            LABEL_STATS.LABEL_UID: None,
            LABEL_STATS.IMAGE_INT: None,
            LABEL_STATS.LABEL: [{LABEL_STATS.PIXEL_PCT: None, LABEL_STATS.IMAGE_INT: None}],
        }

        if self.do_ccp:
            report_format[LABEL_STATS.LABEL][0].update({LABEL_STATS.LABEL_SHAPE: None})
            report_format[LABEL_STATS.LABEL][0].update({LABEL_STATS.LABEL_NCOMP: None})

        super().__init__(report_format)
        self.update_ops(LABEL_STATS.IMAGE_INT, SampleOperations())

        id_seq = ID_SEP_KEY.join([LABEL_STATS.LABEL, "0", LABEL_STATS.IMAGE_INT])
        self.update_ops_nested_label(id_seq, SampleOperations())

    def __call__(self, data):
        ndas = data[self.image_key]  # (1,H,W,D) or (C,H,W,D)
        ndas = [ndas[i] for i in range(ndas.shape[0])]
        ndas_label = data[self.label_key]  # (H,W,D)
        nda_foregrounds = [get_foreground_label(nda, ndas_label) for nda in ndas]
        unique_label = torch.unique(ndas_label).data.cpu().numpy().astype(np.int8).tolist()

        # start = time.time()
        detailed_label_stats = []  # each element is one label
        pixel_sum = 0
        for index in unique_label:
            label_dict: Dict[str, Any] = {}
            mask_index = ndas_label == index

            label_dict[LABEL_STATS.IMAGE_INT] = [
                self.ops[LABEL_STATS.IMAGE_INT].evaluate(nda[mask_index]) for nda in ndas
            ]
            pixel_num = torch.sum(mask_index).data.cpu().numpy()  # pixel_percentage[index]
            label_dict[LABEL_STATS.PIXEL_PCT] = pixel_num.astype(np.float64)
            pixel_sum += pixel_num
            if self.do_ccp:  # apply connected component
                shape_list, ncomponents = get_label_ccp(mask_index)
                label_dict[LABEL_STATS.LABEL_SHAPE] = shape_list
                label_dict[LABEL_STATS.LABEL_NCOMP] = ncomponents

            detailed_label_stats.append(label_dict)
            # logger.debug(f" label {index} stats takes {time.time() - s}")

        # total_percent = np.sum(list(pixel_percentage.values()))
        for i, _ in enumerate(unique_label):
            detailed_label_stats[i][LABEL_STATS.PIXEL_PCT] /= pixel_sum

        analysis = deepcopy(self.get_report_format())
        analysis[LABEL_STATS.LABEL_UID] = unique_label
        analysis[LABEL_STATS.IMAGE_INT] = [self.ops[LABEL_STATS.IMAGE_INT].evaluate(nda_f) for nda_f in nda_foregrounds]
        analysis[LABEL_STATS.LABEL] = detailed_label_stats

        # logger.debug(f"Get label stats spent {time.time()-start}")
        return analysis


class ImageStatsSummaryAnalyzer(Analyzer):
    def __init__(self, case_analyzer_name: str, average: bool = True):
        self.case = case_analyzer_name
        self.summary_average = average
        report_format = {
            IMAGE_STATS.SHAPE: None,
            IMAGE_STATS.CHANNELS: None,
            IMAGE_STATS.CROPPED_SHAPE: None,
            IMAGE_STATS.SPACING: None,
            IMAGE_STATS.INTENSITY: None,
        }
        super().__init__(report_format)

        self.update_ops(IMAGE_STATS.SHAPE, SampleOperations())
        self.update_ops(IMAGE_STATS.CHANNELS, SampleOperations())
        self.update_ops(IMAGE_STATS.CROPPED_SHAPE, SampleOperations())
        self.update_ops(IMAGE_STATS.SPACING, SampleOperations())
        self.update_ops(IMAGE_STATS.INTENSITY, SummaryOperations())

    def __call__(self, data):
        analysis = deepcopy(self.get_report_format())

        axis = 0  # todo: if self.summary_average and data[...].shape > 2, axis = (0, 1)
        analysis[IMAGE_STATS.SHAPE] = self.ops[IMAGE_STATS.SHAPE].evaluate(
            concat_val_to_np(data, [self.case, IMAGE_STATS.SHAPE]), dim=axis
        )
        analysis[IMAGE_STATS.CROPPED_SHAPE] = self.ops[IMAGE_STATS.CROPPED_SHAPE].evaluate(
            concat_val_to_np(data, [self.case, IMAGE_STATS.CROPPED_SHAPE]), dim=axis
        )
        analysis[IMAGE_STATS.SPACING] = self.ops[IMAGE_STATS.SPACING].evaluate(
            concat_val_to_np(data, [self.case, IMAGE_STATS.SPACING]), dim=axis
        )

        axis = None if self.summary_average else 0
        op_keys = analysis[IMAGE_STATS.INTENSITY].keys()  # template, max/min/...
        analysis[IMAGE_STATS.INTENSITY] = self.ops[IMAGE_STATS.INTENSITY].evaluate(
            concat_multikeys_to_dict(data, [self.case, IMAGE_STATS.INTENSITY], op_keys), dim=axis
        )

        return analysis


class FgImageStatsSummaryAnalyzer(Analyzer):
    def __init__(self, case_analyzer_name: str, average=True):
        self.case = case_analyzer_name
        self.summary_average = average

        report_format = {IMAGE_STATS.INTENSITY: None}
        super().__init__(report_format)
        self.update_ops(IMAGE_STATS.INTENSITY, SummaryOperations())

    def __call__(self, data):
        analysis = deepcopy(self.get_report_format())
        axis = None if self.summary_average else 0
        op_keys = analysis[IMAGE_STATS.INTENSITY].keys()  # template, max/min/...
        analysis[IMAGE_STATS.INTENSITY] = self.ops[IMAGE_STATS.INTENSITY].evaluate(
            concat_multikeys_to_dict(data, [self.case, IMAGE_STATS.INTENSITY], op_keys), dim=axis
        )

        return analysis


class LabelStatsSummaryAnalyzer(Analyzer):
    def __init__(self, case_analyzer_name: str, average: bool = True, do_ccp: bool = True):
        self.case = case_analyzer_name
        self.summary_average = average
        self.do_ccp = do_ccp

        report_format = {
            LABEL_STATS.LABEL_UID: None,
            LABEL_STATS.IMAGE_INT: None,
            LABEL_STATS.LABEL: [{LABEL_STATS.PIXEL_PCT: None, LABEL_STATS.IMAGE_INT: None}],
        }
        if self.do_ccp:
            report_format[LABEL_STATS.LABEL][0].update({LABEL_STATS.LABEL_SHAPE: None})
            report_format[LABEL_STATS.LABEL][0].update({LABEL_STATS.LABEL_NCOMP: None})

        super().__init__(report_format)
        self.update_ops(LABEL_STATS.IMAGE_INT, SummaryOperations())

        # label-0-'pixel percentage'
        id_seq = ID_SEP_KEY.join([LABEL_STATS.LABEL, "0", LABEL_STATS.PIXEL_PCT])
        self.update_ops_nested_label(id_seq, SampleOperations())
        # label-0-'image intensity'
        id_seq = ID_SEP_KEY.join([LABEL_STATS.LABEL, "0", LABEL_STATS.IMAGE_INT])
        self.update_ops_nested_label(id_seq, SummaryOperations())
        # label-0-shape
        id_seq = ID_SEP_KEY.join([LABEL_STATS.LABEL, "0", LABEL_STATS.LABEL_SHAPE])
        self.update_ops_nested_label(id_seq, SampleOperations())
        # label-0-ncomponents
        id_seq = ID_SEP_KEY.join([LABEL_STATS.LABEL, "0", LABEL_STATS.LABEL_NCOMP])
        self.update_ops_nested_label(id_seq, SampleOperations())

    def __call__(self, data):
        analysis = deepcopy(self.get_report_format())
        unique_label = label_union(concat_val_to_np(data, [self.case, LABEL_STATS.LABEL_UID], flatten=True))

        axis = None if self.summary_average else 0
        analysis[LABEL_STATS.LABEL_UID] = unique_label
        op_keys = analysis[LABEL_STATS.IMAGE_INT].keys()  # template, max/min/...
        analysis[LABEL_STATS.IMAGE_INT] = self.ops[LABEL_STATS.IMAGE_INT].evaluate(
            concat_multikeys_to_dict(data, [self.case, LABEL_STATS.IMAGE_INT], op_keys), dim=axis
        )

        detailed_label_list = []

        for label_id in unique_label:
            stats = {}
            axis = 0  # todo: if self.summary_average and data[...].shape > 2, axis = (0, 1)
            for key in [LABEL_STATS.PIXEL_PCT, LABEL_STATS.LABEL_SHAPE, LABEL_STATS.LABEL_NCOMP]:
                stats[key] = self.ops[LABEL_STATS.LABEL][0][key].evaluate(
                    concat_val_to_np(
                        data, [self.case, LABEL_STATS.LABEL, label_id, key], allow_missing=True, flatten=True
                    ),
                    dim=axis,
                )

            axis = None
            label_image_intensity = [self.case, LABEL_STATS.LABEL, label_id, LABEL_STATS.IMAGE_INT]
            op_keys = analysis[LABEL_STATS.LABEL][0][LABEL_STATS.IMAGE_INT].keys()
            stats[LABEL_STATS.IMAGE_INT] = self.ops[LABEL_STATS.LABEL][0][LABEL_STATS.IMAGE_INT].evaluate(
                concat_multikeys_to_dict(data, label_image_intensity, op_keys, allow_missing=True, flatten=True),
                dim=axis,
            )

            detailed_label_list.append(stats)

        analysis[LABEL_STATS.LABEL] = detailed_label_list

        return analysis
