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
from typing import Any, Dict, List, Optional, Type

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
from monai.data.meta_tensor import MetaTensor
from monai.transforms.transform import MapTransform
from monai.transforms.utils_pytorch_numpy_unification import unique, sum
from monai.utils.enums import IMAGE_STATS, LABEL_STATS
from monai.utils.misc import label_union


class Analyzer(MapTransform, ABC):
    """
    The Analyzer component is a base class. Other classes inherit this class will provide a callable
    with the same class name and produces one pre-formatted dictionary for the input data. The format
    is pre-defined by the init function of the class that inherit this base class. Function operations
    can also be registerred before the runtime of the callable.

    Args:
        report_format: a dictionary that outlines the key structures of the report format.

    """
    def __init__(self, report_format: dict) -> None:
        super().__init__(None)
        self.report_format = report_format
        self.ops = ConfigParser({})

    def update_ops(self, key: str, op: Type[Operations]):
        """
        Register an statistical operation to the Analyzer and update the report_format

        Args:
            key: value key in the report.
            op: Operation sub-class object that represents statistical operations.

        """
        self.ops[key] = op
        parser = ConfigParser(self.report_format)

        if parser.get(key, "NA") != "NA":
            parser[key] = op

        self.report_format = parser.config

    def update_ops_nested_label(self, nested_key: str, op: Type[Operations]):
        """
        Update operations for nested label format. Operation value in report_format will be resolved
        to a dict with only keys

        Args:
            nested_key: str that has format of 'key1#0#key2'.
            op: Operation sub-class object that represents statistical operations.
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
        Get the report format by resolving the registered operations recursively.

        Returns:
            a dictionary with {keys: None} pairs.

        """
        self.resolve_format(self.report_format)
        return self.report_format

    @staticmethod
    def unwrap_ops(func: Type[Operations]):
        """
        Unwrap a function value and generates the same set keys in a dict when the function is actually
        called in runtime

        Args:
            func: Operation sub-class object that represents statistical operations.

        Returns:
            a dict with a set of keys.

        """
        ret = dict.fromkeys([key for key in func.data])
        if hasattr(func, "data_addon"):
            for key in func.data_addon:
                ret.update({key: None})
        return ret

    def resolve_format(self, report: dict):
        """
        Resolve the format of the pre-defined report.

        Args:
            report: the dictionary to resolve. Values will be replaced in-place.

        """
        for k, v in report.items():
            if issubclass(v.__class__, Operations):
                report[k] = self.unwrap_ops(v)
            elif isinstance(v, list) and len(v) > 0:
                self.resolve_format(v[0])
            else:
                report[k] = v

    @abstractmethod
    def __call__(self, data: Any):
        """Analyze the dict format dataset, return the summary report"""
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class ImageStatsCaseAnalyzer(Analyzer):
    """
    Analyzer to extract image stats properties for each case(image). 

    Args:
        image_key: the key to find image data in the callable function input (data) 
        meta_key_postfix: the postfix to append for meta_dict ("image_meta_dict")
    
    Examples:

    .. code-block:: python

        import numpy as np
        from monai.auto3dseg.analyzer import ImageStatsCaseAnalyzer

        input = {}
        input['image'] = np.random.rand(1,30,30,30)
        input['image_meta_dict'] = {'affine': np.eye(4)}
        analyzer = ImageStatsCaseAnalyzer(image_key="image")
        print(analyzer(input))

    """
    def __init__(self, image_key: str, meta_key_postfix: Optional[str] = "_meta_dict"):

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
        """
        Callable to execute the pre-defined functions

        Returns:
            A dictionary. The dict has the key in self.report_format. The value of 
            IMAGE_STATS.INTENSITY is in a list format. Each element of the value list 
            has stats pre-defined by SampleOperations (max, min, ....)
        
        Note:
            The stats operation uses numpy and torch to compute max, min, and other
            functions. If the input has nan/inf, the stats results will be nan/inf.
        
        """
        # from time import time
        # start = time.time()
        ndas = data[self.image_key]
        ndas = [ndas[i] for i in range(ndas.shape[0])]
        if "nda_croppeds" not in data:
            data["nda_croppeds"] = [get_foreground_image(nda) for nda in ndas]
        nda_croppeds = data["nda_croppeds"]

        # perform calculation
        report = deepcopy(self.get_report_format())

        report[IMAGE_STATS.SHAPE] = [list(nda.shape) for nda in ndas]
        report[IMAGE_STATS.CHANNELS] = len(ndas)
        report[IMAGE_STATS.CROPPED_SHAPE] = [list(nda_c.shape) for nda_c in nda_croppeds]
        report[IMAGE_STATS.SPACING] = np.tile(
            np.diag(data[self.image_meta_key]["affine"])[:3], [len(ndas), 1]
        ).tolist()
        report[IMAGE_STATS.INTENSITY] = [self.ops[IMAGE_STATS.INTENSITY].evaluate(nda_c) for nda_c in nda_croppeds]

        # logger.debug(f"Get image stats spent {time.time()-start}")
        return report


class FgImageStatsCaseAnalyzer(Analyzer):
    """
    Analyzer to extract foreground label properties for each case(image and label). 

    Args:
        image_key: the key to find image data in the callable function input (data) 
        label_key: the key to find label data in the callable function input (data)

    Examples:

    .. code-block:: python

        import numpy as np
        from monai.auto3dseg.analyzer import FgImageStatsCaseAnalyzer

        input = {}
        input['image'] = np.random.rand(1,30,30,30)
        input['label'] = np.ones([30,30,30])
        analyzer = FgImageStatsCaseAnalyzer(image_key='image', label_key='label')
        print(analyzer(input))
    
    """
    def __init__(self, image_key, label_key):

        self.image_key = image_key
        self.label_key = label_key

        report_format = {IMAGE_STATS.INTENSITY: None}

        super().__init__(report_format)
        self.update_ops(IMAGE_STATS.INTENSITY, SampleOperations())

    def __call__(self, data) -> dict:
        """
        Callable to execute the pre-defined functions

        Returns:
            A dictionary. The dict has the key in self.report_format and value
            in a list format. Each element of the value list has stats pre-defined
            by SampleOperations (max, min, ....)
        
        Note:
            The stats operation uses numpy and torch to compute max, min, and other
            functions. If the input has nan/inf, the stats results will be nan/inf.
        """

        ndas = data[self.image_key]  # (1,H,W,D) or (C,H,W,D)
        ndas = [ndas[i] for i in range(ndas.shape[0])]
        ndas_label = data[self.label_key]  # (H,W,D)
        nda_foregrounds = [get_foreground_label(nda, ndas_label) for nda in ndas]

        # perform calculation
        report = deepcopy(self.get_report_format())

        report[IMAGE_STATS.INTENSITY] = [self.ops[IMAGE_STATS.INTENSITY].evaluate(nda_f) for nda_f in nda_foregrounds]
        return report


class LabelStatsCaseAnalyzer(Analyzer):
    """
    Analyzer to extract label stats properties for each case(image and label). 

    Args:
        image_key: the key to find image data in the callable function input (data) 
        label_key: the key to find label data in the callable function input (data) 
        do_ccp: performs connected component analysis. Default is True.
    
    Examples:

    .. code-block:: python

        import numpy as np
        from monai.auto3dseg.analyzer import LabelStatsCaseAnalyzer

        input = {}
        input['image'] = np.random.rand(1,30,30,30)
        input['label'] = np.ones([30,30,30])
        analyzer = LabelStatsCaseAnalyzer(image_key='image', label_key='label')
        print(analyzer(input))
    
    """
    def __init__(self, image_key: str, label_key: str, do_ccp: Optional[bool] = True):

        self.image_key = image_key
        self.label_key = label_key
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
        """
        Callable to execute the pre-defined functions

        Returns:
            A dictionary. The dict has the key in self.report_format and value
            in a list format. Each element of the value list has stats pre-defined
            by SampleOperations (max, min, ....)
        
        Examples:
            output dict contains {
                LABEL_STATS.LABEL_UID:[0,1,3], 
                LABEL_STATS.IMAGE_INT: {...},
                LABEL_STATS.LABEL:[
                    {
                        LABEL_STATS.PIXEL_PCT: 0.8,
                        LABEL_STATS.IMAGE_INT: {...},
                        LABEL_STATS.LABEL_SHAPE: [...],
                        LABEL_STATS.LABEL_NCOMP: 1
                    }
                    {
                        LABEL_STATS.PIXEL_PCT: 0.1,
                        LABEL_STATS.IMAGE_INT: {...},
                        LABEL_STATS.LABEL_SHAPE: [...],
                        LABEL_STATS.LABEL_NCOMP: 1
                    }
                    {
                        LABEL_STATS.PIXEL_PCT: 0.1,
                        LABEL_STATS.IMAGE_INT: {...},
                        LABEL_STATS.LABEL_SHAPE: [...],
                        LABEL_STATS.LABEL_NCOMP: 1
                    }
                ]
                }
        
        Notes:
            The label class_ID of the dictionary in LABEL_STATS.LABEL IS NOT the 
            index. Instead, the class_ID is the LABEL_STATS.LABEL_UID with the same
            index. For instance, the last dict in LABEL_STATS.LABEL in the Examples
            is 3, which is the last element under LABEL_STATS.LABEL_UID.

            The stats operation uses numpy and torch to compute max, min, and other
            functions. If the input has nan/inf, the stats results will be nan/inf.
        """
        ndas = data[self.image_key]  # (1,H,W,D) or (C,H,W,D)
        ndas = [ndas[i] for i in range(ndas.shape[0])]
        ndas_label = data[self.label_key]  # (H,W,D)
        nda_foregrounds = [get_foreground_label(nda, ndas_label) for nda in ndas]

        unique_label = unique(ndas_label)
        if isinstance(ndas_label, (MetaTensor, torch.Tensor)):
            unique_label = unique_label.data.cpu().numpy()
 
        unique_label = unique_label.astype(np.int8).tolist()

        # start = time.time()
        detailed_label_stats = []  # each element is one label
        pixel_sum = 0
        for index in unique_label:
            label_dict: Dict[str, Any] = {}
            mask_index = ndas_label == index

            label_dict[LABEL_STATS.IMAGE_INT] = [
                self.ops[LABEL_STATS.IMAGE_INT].evaluate(nda[mask_index]) for nda in ndas
            ]
            pixel_num = sum(mask_index)
            if isinstance(pixel_num, (MetaTensor, torch.Tensor)):
                pixel_num = pixel_num.data.cpu().numpy()  # pixel_percentage[index]
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

        report = deepcopy(self.get_report_format())
        report[LABEL_STATS.LABEL_UID] = unique_label
        report[LABEL_STATS.IMAGE_INT] = [self.ops[LABEL_STATS.IMAGE_INT].evaluate(nda_f) for nda_f in nda_foregrounds]
        report[LABEL_STATS.LABEL] = detailed_label_stats

        # logger.debug(f"Get label stats spent {time.time()-start}")
        return report


class ImageStatsSummaryAnalyzer(Analyzer):
    """
    Analyzer to process the values of specific key `key_in_case` in a list of dict. 
    Typically, the list of dict is the output of case analyzer under the same prefix
    (ImageStatsCaseAnalyzer).

    Args:
        key_in_case: the key of the to-process value in the dict
        average: whether to average the statistical value across different image modalities.

    """
    def __init__(self, key_in_case: str, average: bool = True):
        self.key_case = key_in_case
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

    def __call__(self, data: List[Dict]):
        """
        Callable to execute the pre-defined functions

        Returns:
            A dictionary. The dict has the key in self.report_format and value
            in a list format. Each element of the value list has stats pre-defined
            by SampleOperations (max, min, ....)
        
        Examples:
            output dict contains a dictionary for all of the following keys{
                IMAGE_STATS.SHAPE:{...} 
                IMAGE_STATS.CHANNELS: {...},
                IMAGE_STATS.CROPPED_SHAPE: {...},
                IMAGE_STATS.SPACING: {...},
                IMAGE_STATS.INTENSITY: {...},
                }

        Notes:
            The stats operation uses numpy and torch to compute max, min, and other
            functions. If the input has nan/inf, the stats results will be nan/inf.
        """
        if not isinstance(data, list):
            return ValueError(f"Callable {self.__class__} requires list inputs")

        report = deepcopy(self.get_report_format())

        shape_np = concat_val_to_np(data, [self.key_case, IMAGE_STATS.SHAPE])
        crops_np = concat_val_to_np(data, [self.key_case, IMAGE_STATS.CROPPED_SHAPE])
        space_np = concat_val_to_np(data, [self.key_case, IMAGE_STATS.SPACING])

        op_keys = report[IMAGE_STATS.INTENSITY].keys()  # template, max/min/...
        intst_dict = concat_multikeys_to_dict(data, [self.key_case, IMAGE_STATS.INTENSITY], op_keys)

        report[IMAGE_STATS.SHAPE] = self.ops[IMAGE_STATS.SHAPE].evaluate(shape_np, 
            dim=(0, 1) if shape_np.ndim > 2 and self.summary_average else 0)
        report[IMAGE_STATS.CROPPED_SHAPE] = self.ops[IMAGE_STATS.CROPPED_SHAPE].evaluate(crops_np, 
            dim=(0, 1) if crops_np.ndim > 2 and self.summary_average else 0)
        report[IMAGE_STATS.SPACING] = self.ops[IMAGE_STATS.SPACING].evaluate(space_np, 
            dim=(0, 1) if space_np.ndim > 2 and self.summary_average else 0)
        report[IMAGE_STATS.INTENSITY] = self.ops[IMAGE_STATS.INTENSITY].evaluate(intst_dict,
            dim=None if self.summary_average else 0)

        return report


class FgImageStatsSummaryAnalyzer(Analyzer):
    def __init__(self, key_in_case: str, average=True):
        self.key_case = key_in_case
        self.summary_average = average

        report_format = {IMAGE_STATS.INTENSITY: None}
        super().__init__(report_format)
        self.update_ops(IMAGE_STATS.INTENSITY, SummaryOperations())

    def __call__(self, data: List[Dict]):
        if not isinstance(data, list):
            return ValueError(f"Callable {self.__class__} requires list inputs")

        report = deepcopy(self.get_report_format())
        op_keys = report[IMAGE_STATS.INTENSITY].keys()  # template, max/min/...
        intst_dict = concat_multikeys_to_dict(data, [self.key_case, IMAGE_STATS.INTENSITY], op_keys)

        report[IMAGE_STATS.INTENSITY] = self.ops[IMAGE_STATS.INTENSITY].evaluate(intst_dict, 
            dim=None if self.summary_average else 0)

        return report


class LabelStatsSummaryAnalyzer(Analyzer):
    def __init__(self, key_in_case: str, average: bool = True, do_ccp: bool = True):
        self.key_case = key_in_case
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

    def __call__(self, data: List[Dict]):
        if not isinstance(data, list):
            return ValueError(f"Callable {self.__class__} requires list inputs")

        report = deepcopy(self.get_report_format())
        uid_np = concat_val_to_np(data, [self.key_case, LABEL_STATS.LABEL_UID], flatten=True)
        unique_label = label_union(uid_np)

        report[LABEL_STATS.LABEL_UID] = unique_label
        op_keys = report[LABEL_STATS.IMAGE_INT].keys()  # template, max/min/...
        intst_dict = concat_multikeys_to_dict(data, [self.key_case, LABEL_STATS.IMAGE_INT], op_keys)
        report[LABEL_STATS.IMAGE_INT] = self.ops[LABEL_STATS.IMAGE_INT].evaluate(intst_dict,
            dim=None if self.summary_average else 0)

        detailed_label_list = []

        for label_id in unique_label:
            stats = {}
            axis = 0  # todo: if self.summary_average and data[...].shape > 2, axis = (0, 1)
            for k in [LABEL_STATS.PIXEL_PCT, LABEL_STATS.LABEL_SHAPE, LABEL_STATS.LABEL_NCOMP]:
                v_np = concat_val_to_np(data, [self.key_case, LABEL_STATS.LABEL, label_id, k], 
                    allow_missing=True, flatten=True)
                stats[k] = self.ops[LABEL_STATS.LABEL][0][k].evaluate(v_np,
                    dim=(0, 1) if v_np.ndim > 2 and self.summary_average else 0)

            intst_fixed_keys = [self.key_case, LABEL_STATS.LABEL, label_id, LABEL_STATS.IMAGE_INT]
            op_keys = report[LABEL_STATS.LABEL][0][LABEL_STATS.IMAGE_INT].keys()
            intst_dict = concat_multikeys_to_dict(data, intst_fixed_keys, op_keys, allow_missing=True, flatten=True)
            stats[LABEL_STATS.IMAGE_INT] = self.ops[LABEL_STATS.LABEL][0][LABEL_STATS.IMAGE_INT].evaluate(intst_dict,
                    dim=None if self.summary_average else 0)

            detailed_label_list.append(stats)

        report[LABEL_STATS.LABEL] = detailed_label_list

        return report
