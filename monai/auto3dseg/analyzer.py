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

import numpy as np
import torch

from abc import abstractmethod, ABC
from copy import deepcopy

from monai.transforms import transform
from monai.utils.misc import label_union
from monai.utils.enums import IMAGE_STATS, LABEL_STATS
from monai.auto3dseg.utils import get_foreground_image, get_foreground_label, get_label_ccp, concat_val_to_np
from monai.auto3dseg.operations import Operations, SampleOperations, SummaryOperations

from typing import Any, Dict

class Analyzer(transform.MapTransform, ABC):
    def __init__(self, report_format):
        self.report_format = report_format
        self.ops = {}

    def update_ops(self, key, op):
        """
        """
        self.ops[key] = op

        if key in self.report_format:
            self.report_format[key] = op  # value in report_format will be resolved to a dict with only keys

    def resolve_ops(self, func):
        ret = dict.fromkeys([key for key in func.data])
        if hasattr(func, 'data_addon'):
            for key in func.data_addon:
                ret.update({key: None})
        return ret

    def get_report_format(self):
        for k, v in self.report_format.items():
            if issubclass(v.__class__, Operations):
                self.report_format[k] = self.resolve_ops(v)
            else: 
                self.report_format[k] = v
        
        return self.report_format
    
    @abstractmethod
    def __call__(self, data):
        """Analyze the dict format dataset, return the summary report"""
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

class ImageStatsCaseAnalyzer(Analyzer):
    def __init__(self, image_key, label_key, meta_key_postfix = "_meta_dict"):

        self.image_key = image_key
        self.label_key = label_key
        self.image_meta_key = self.image_key + meta_key_postfix
        self.label_meta_key = self.label_key + meta_key_postfix

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
        analysis[IMAGE_STATS.SPACING] = np.tile(np.diag(data[self.image_meta_key]["affine"])[:3], [len(ndas), 1]).tolist()
        analysis[IMAGE_STATS.INTENSITY] = [self.ops[IMAGE_STATS.INTENSITY].evaluate(nda_c) for nda_c in nda_croppeds]

        # logger.debug(f"Get image stats spent {time.time()-start}")
        return analysis

class FgImageStatsCasesAnalyzer(Analyzer):
    def __init__(self, image_key, label_key, meta_key_postfix = "_meta_dict"):

        self.image_key = image_key
        self.label_key = label_key
        self.image_meta_key = self.image_key + meta_key_postfix
        self.label_meta_key = self.label_key + meta_key_postfix

        report_format = {
            IMAGE_STATS.INTENSITY: None
        }

        super().__init__(report_format)
        self.update_ops(IMAGE_STATS.INTENSITY, SampleOperations())
    
    def __call__(self, data):

        ndas = data[self.image_key] # (1,H,W,D) or (C,H,W,D)
        ndas = [ndas[i] for i in range(ndas.shape[0])]
        ndas_label = data[self.label_key] # (H,W,D)
        nda_foregrounds = [get_foreground_label(nda, ndas_label) for nda in ndas]

        # perform calculation
        analysis = deepcopy(self.get_report_format())

        analysis[IMAGE_STATS.INTENSITY] = [self.ops[IMAGE_STATS.INTENSITY].evaluate(nda_f) for nda_f in nda_foregrounds]
        return analysis

class LabelStatsCaseAnalyzer(Analyzer):
    def __init__(self, image_key, label_key, meta_key_postfix = "_meta_dict", do_ccp: bool = True):

        self.image_key = image_key
        self.label_key = label_key
        self.image_meta_key = self.image_key + meta_key_postfix
        self.label_meta_key = self.label_key + meta_key_postfix
        self.do_ccp = do_ccp

        report_format = {
            LABEL_STATS.LABEL_UID: None,
            LABEL_STATS.PIXEL_PCT: None,
            LABEL_STATS.IMAGE_INT: None,
            LABEL_STATS.LABEL: [{
                LABEL_STATS.IMAGE_INT: None,
                LABEL_STATS.LABEL_SHAPE: None,
                LABEL_STATS.LABEL_NCOMP: None,
            }],
        }

        super().__init__(report_format)
        self.update_ops(LABEL_STATS.IMAGE_INT, SampleOperations())
        self.update_ops_label_list(LABEL_STATS.LABEL, SampleOperations())

    def update_ops_label_list(self, key, op):
        self.ops[key] = op
        # todo: add support for the list-type item print-out
    
    def __call__(self, data):
        ndas = data[self.image_key] # (1,H,W,D) or (C,H,W,D)
        ndas = [ndas[i] for i in range(ndas.shape[0])]
        ndas_label = data[self.label_key] # (H,W,D)
        nda_foregrounds = [get_foreground_label(nda, ndas_label) for nda in ndas]
        unique_label = torch.unique(ndas_label).data.cpu().numpy().astype(np.int8).tolist()

        # start = time.time()
        label_stats = []
        pixel_percentage = {}
        for index in unique_label:
            label_dict: Dict[str, Any] = {}
            mask_index = ndas_label == index
            label_dict[LABEL_STATS.IMAGE_INT] = [self.ops[LABEL_STATS.IMAGE_INT].evaluate(nda[mask_index]) for nda in ndas]
            # logger.debug(f" label {index} stats takes {time.time() - s}")
            pixel_percentage[index] = torch.sum(mask_index).data.cpu().numpy()
            if self.do_ccp:  # apply connected component
                shape_list, ncomponents = get_label_ccp(mask_index)
                label_dict[LABEL_STATS.LABEL_SHAPE] = shape_list
                label_dict[LABEL_STATS.LABEL_NCOMP] = ncomponents
            
            label_stats.append(label_dict)

        total_percent = np.sum(list(pixel_percentage.values()))
        for key, value in pixel_percentage.items():
            pixel_percentage[key] = float(value / total_percent)
        
        analysis = deepcopy(self.get_report_format())
        analysis[LABEL_STATS.LABEL_UID] = unique_label
        analysis[LABEL_STATS.IMAGE_INT] = [self.ops[LABEL_STATS.IMAGE_INT].evaluate(nda_f) for nda_f in nda_foregrounds]
        analysis[LABEL_STATS.LABEL] = label_stats
        analysis[LABEL_STATS.PIXEL_PCT] = pixel_percentage
        
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
            IMAGE_STATS.INTENSITY: None
        }
        super().__init__(report_format)

        self.update_ops(IMAGE_STATS.SHAPE, SampleOperations())
        self.update_ops(IMAGE_STATS.CHANNELS, SampleOperations())
        self.update_ops(IMAGE_STATS.CROPPED_SHAPE, SampleOperations())
        self.update_ops(IMAGE_STATS.SPACING, SampleOperations())
        self.update_ops(IMAGE_STATS.INTENSITY, SummaryOperations())
    
    
    def concat_to_dict(self, key: str, ld_data):
        """
        Pinpointing the key in data structure: list of dicts and concat the value
        """
        values = [d[self.case][key] for d in ld_data]  # ld: list of dicts
        # analysis is a list of list
        key_values = {}
        for k in values[0][0]:
            key_values[k] = np.concatenate([[val[0][k].cpu().numpy() for val in values]]) #gpu/cpu issue
        
        return key_values

    def __call__(self, data):
        analysis = deepcopy(self.get_report_format())
        
        axis = 0 # todo: if self.summary_average and data[...].shape > 2, axis = (0, 1)
        analysis[IMAGE_STATS.SHAPE] = self.ops[IMAGE_STATS.SHAPE].evaluate(
            concat_val_to_np(data, [self.case, IMAGE_STATS.SHAPE]), dim=axis)
        analysis[IMAGE_STATS.CROPPED_SHAPE] = self.ops[IMAGE_STATS.CROPPED_SHAPE].evaluate(
            concat_val_to_np(data, [self.case, IMAGE_STATS.CROPPED_SHAPE]), dim=axis)
        analysis[IMAGE_STATS.SPACING] = self.ops[IMAGE_STATS.SPACING].evaluate(
            concat_val_to_np(data, [self.case, IMAGE_STATS.SPACING]), dim=axis)

        axis = None if self.summary_average else 0
        analysis[IMAGE_STATS.INTENSITY] = self.ops[IMAGE_STATS.INTENSITY].evaluate(
            self.concat_to_dict(IMAGE_STATS.INTENSITY, data), dim=axis)

        return analysis

class FgImageStatsSummaryAnalyzer(Analyzer):
    def __init__(self, case_analyzer_name: str, average=True):
        self.case = case_analyzer_name
        self.summary_average = average

        report_format = {
            IMAGE_STATS.INTENSITY: None
        }
        super().__init__(report_format)
        self.update_ops(IMAGE_STATS.INTENSITY, SummaryOperations())
    
    def concat_to_dict(self, key: str, ld_data):
        """
        Pinpointing the key in data structure: list of dicts and concat the value
        """
        values = [d[self.case][key] for d in ld_data]  # ld: list of dicts
        # analysis is a list of list
        key_values = {}
        for k in values[0][0]:
            key_values[k] = np.concatenate([[val[0][k].cpu().numpy() for val in values]]) #gpu/cpu issue
        
        return key_values
    
    def __call__(self, data):
        analysis = deepcopy(self.get_report_format())
        axis = None if self.summary_average else 0
        analysis[IMAGE_STATS.INTENSITY] = self.ops[IMAGE_STATS.INTENSITY].evaluate(
            self.concat_to_dict(IMAGE_STATS.INTENSITY, data), dim=axis)
        
        return analysis

class LabelStatsSummaryAnalyzer(Analyzer):
    def __init__(self, case_analyzer_name: str, average: bool = True, do_ccp: bool = True):
        self.case = case_analyzer_name
        self.summary_average = average
        self.do_ccp = do_ccp

        report_format = {
            LABEL_STATS.LABEL_UID: None,
            LABEL_STATS.PIXEL_PCT: None,
            LABEL_STATS.IMAGE_INT: None,
            LABEL_STATS.LABEL: [{
                LABEL_STATS.IMAGE_INT: None,
                LABEL_STATS.LABEL_SHAPE: None,
                LABEL_STATS.LABEL_NCOMP: None,
            }]
        }
        super().__init__(report_format)
        self.update_ops(LABEL_STATS.IMAGE_INT, SummaryOperations())
        self.update_ops(LABEL_STATS.LABEL_SHAPE, SampleOperations())
        self.update_ops(LABEL_STATS.LABEL_NCOMP, SampleOperations())

    def concat_label_np(self, label_id, key: str, data):
        values = [d[self.case][key] for d in data]
        # analysis is a list of list
        return np.concatenate([[val[label_id] for val in values if label_id in val]]) #gpu/cpu issue

    def concat_to_dict(self, key: str, data):
        """
        Pinpointing the key in data structure: list of dicts and concat the value
        """
        values = [d[self.case][key] for d in data]
        # analysis is a list of list
        key_values = {}
        for k in values[0][0]:
            key_values[k] = np.concatenate([[val[0][k].cpu().numpy() for val in values]]) #gpu/cpu issue
        
        return key_values

    def concat_label_to_dict(self, label_id: int, key: str, data):
        values = []
        for d in data:
            if label_id in d[self.case][LABEL_STATS.LABEL_UID]:
                idx = d[self.case][LABEL_STATS.LABEL_UID].index(label_id)
                values.append(d[self.case][LABEL_STATS.LABEL][idx][key])

        if isinstance(values[0], list):
            if isinstance(values[0][0], list):
                return np.concatenate([[np.array(v[0]) for v in values]])
            elif isinstance(values[0][0], dict):
                key_values = {}
                for k in values[0][0]:
                    key_values[k] = np.concatenate([[val[0][k].cpu().numpy() for val in values]]) #gpu/cpu issue
                return key_values
            else:
                raise NotImplementedError("The method to get number is not implemented. Unable to find the values.")
        else:
            return np.array(values)


    def __call__(self, data):
        analysis = deepcopy(self.get_report_format())
        
        unique_label = label_union(concat_val_to_np(data, [self.case, LABEL_STATS.LABEL_UID], flatten=True))
        
        pixel_summary = {}
        for label_id in unique_label:
            pixel_summary.update({label_id: self.concat_label_np(label_id, LABEL_STATS.PIXEL_PCT, data)})
        
        axis = None if self.summary_average else 0

        analysis[LABEL_STATS.LABEL_UID] = unique_label
        analysis[LABEL_STATS.PIXEL_PCT] = [{k: np.mean(v)} for k, v in pixel_summary.items()]
        analysis[LABEL_STATS.IMAGE_INT] = self.ops[LABEL_STATS.IMAGE_INT].evaluate(
            self.concat_to_dict(LABEL_STATS.IMAGE_INT, data), dim=axis)
        
        analysis[LABEL_STATS.LABEL] = []
        for label_id in unique_label:
            stats = {}
            for key in [LABEL_STATS.IMAGE_INT, LABEL_STATS.LABEL_SHAPE, LABEL_STATS.LABEL_NCOMP]:
                stats[key] = self.ops[key].evaluate(self.concat_label_to_dict(label_id, key, data), dim=axis)
            analysis[LABEL_STATS.LABEL].append(stats)
            
        return analysis
