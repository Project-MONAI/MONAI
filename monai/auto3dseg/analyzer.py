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

import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from monai.apps.utils import get_logger
from monai.auto3dseg.operations import Operations, SampleOperations, SummaryOperations
from monai.auto3dseg.utils import (
    concat_multikeys_to_dict,
    concat_val_to_np,
    get_foreground_image,
    get_foreground_label,
    get_label_ccp,
    verify_report_format,
)
from monai.bundle.config_parser import ConfigParser
from monai.bundle.utils import ID_SEP_KEY
from monai.data import MetaTensor, affine_to_spacing
from monai.transforms.transform import MapTransform
from monai.transforms.utils_pytorch_numpy_unification import sum, unique
from monai.utils import convert_to_numpy
from monai.utils.enums import DataStatsKeys, ImageStatsKeys, LabelStatsKeys
from monai.utils.misc import ImageMetaKey, label_union

logger = get_logger(module_name=__name__)

__all__ = [
    "Analyzer",
    "ImageStats",
    "FgImageStats",
    "LabelStats",
    "ImageStatsSumm",
    "FgImageStatsSumm",
    "LabelStatsSumm",
    "FilenameStats",
    "ImageHistogram",
    "ImageHistogramSumm",
]


class Analyzer(MapTransform, ABC):
    """
    The Analyzer component is a base class. Other classes inherit this class will provide a callable
    with the same class name and produces one pre-formatted dictionary for the input data. The format
    is pre-defined by the init function of the class that inherit this base class. Function operations
    can also be registered before the runtime of the callable.

    Args:
        report_format: a dictionary that outlines the key structures of the report format.

    """

    def __init__(self, stats_name: str, report_format: dict) -> None:
        super().__init__(None)
        parser = ConfigParser(report_format, globals=False)  # ConfigParser.globals not picklable
        self.report_format = parser.get("")
        self.stats_name = stats_name
        self.ops = ConfigParser({}, globals=False)

    def update_ops(self, key: str, op):
        """
        Register a statistical operation to the Analyzer and update the report_format.

        Args:
            key: value key in the report.
            op: Operation sub-class object that represents statistical operations.

        """
        self.ops[key] = op
        parser = ConfigParser(self.report_format)

        if parser.get(key, "None") != "None":
            parser[key] = op

        self.report_format = parser.get("")

    def update_ops_nested_label(self, nested_key: str, op):
        """
        Update operations for nested label format. Operation value in report_format will be resolved
        to a dict with only keys.

        Args:
            nested_key: str that has format of 'key1#0#key2'.
            op: Operation sub-class object that represents statistical operations.
        """
        keys = nested_key.split(ID_SEP_KEY)
        if len(keys) != 3:
            raise ValueError("Nested_key input format is wrong. Please ensure it is like key1#0#key2")
        root: str
        child_key: str
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
    def unwrap_ops(func):
        """
        Unwrap a function value and generates the same set keys in a dict when the function is actually
        called in runtime

        Args:
            func: Operation sub-class object that represents statistical operations. The func object
                should have a `data` dictionary which stores the statistical operation information.
                For some operations (ImageStats for example), it may also contain the data_addon
                property, which is part of the update process.

        Returns:
            a dict with a set of keys.

        """
        ret = dict.fromkeys(list(func.data))
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
            if isinstance(v, Operations):
                report[k] = self.unwrap_ops(v)
            elif isinstance(v, list) and len(v) > 0:
                self.resolve_format(v[0])
            else:
                report[k] = v

    @abstractmethod
    def __call__(self, data: Any):
        """Analyze the dict format dataset, return the summary report"""
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class ImageStats(Analyzer):
    """
    Analyzer to extract image stats properties for each case(image).

    Args:
        image_key: the key to find image data in the callable function input (data)

    Examples:

    .. code-block:: python

        import numpy as np
        from monai.auto3dseg import ImageStats
        from monai.data import MetaTensor

        input = {}
        input['image'] = np.random.rand(1,30,30,30)
        input['image'] = MetaTensor(np.random.rand(1,30,30,30))  # MetaTensor
        analyzer = ImageStats(image_key="image")
        print(analyzer(input)["image_stats"])

    Notes:
        if the image data is NumPy array, the spacing stats will be [1.0] * `ndims` of the array,
        where the `ndims` is the lesser value between the image dimension and 3.

    """

    def __init__(self, image_key: str, stats_name: str = "image_stats") -> None:

        if not isinstance(image_key, str):
            raise ValueError("image_key input must be str")

        self.image_key = image_key

        report_format = {
            ImageStatsKeys.SHAPE: None,
            ImageStatsKeys.CHANNELS: None,
            ImageStatsKeys.CROPPED_SHAPE: None,
            ImageStatsKeys.SPACING: None,
            ImageStatsKeys.INTENSITY: None,
        }

        super().__init__(stats_name, report_format)
        self.update_ops(ImageStatsKeys.INTENSITY, SampleOperations())

    def __call__(self, data):
        """
        Callable to execute the pre-defined functions

        Returns:
            A dictionary. The dict has the key in self.report_format. The value of
            ImageStatsKeys.INTENSITY is in a list format. Each element of the value list
            has stats pre-defined by SampleOperations (max, min, ....).

        Raises:
            RuntimeError if the stats report generated is not consistent with the pre-
                defined report_format.

        Note:
            The stats operation uses numpy and torch to compute max, min, and other
            functions. If the input has nan/inf, the stats results will be nan/inf.

        """
        d = dict(data)
        start = time.time()
        restore_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        ndas = [d[self.image_key][i] for i in range(d[self.image_key].shape[0])]
        if "nda_croppeds" not in d:
            nda_croppeds = [get_foreground_image(nda) for nda in ndas]

        # perform calculation
        report = deepcopy(self.get_report_format())

        report[ImageStatsKeys.SHAPE] = [list(nda.shape) for nda in ndas]
        report[ImageStatsKeys.CHANNELS] = len(ndas)
        report[ImageStatsKeys.CROPPED_SHAPE] = [list(nda_c.shape) for nda_c in nda_croppeds]
        report[ImageStatsKeys.SPACING] = (
            affine_to_spacing(data[self.image_key].affine).tolist()
            if isinstance(data[self.image_key], MetaTensor)
            else [1.0] * min(3, data[self.image_key].ndim)
        )
        report[ImageStatsKeys.INTENSITY] = [
            self.ops[ImageStatsKeys.INTENSITY].evaluate(nda_c) for nda_c in nda_croppeds
        ]

        if not verify_report_format(report, self.get_report_format()):
            raise RuntimeError(f"report generated by {self.__class__} differs from the report format.")

        d[self.stats_name] = report

        torch.set_grad_enabled(restore_grad_state)
        logger.debug(f"Get image stats spent {time.time()-start}")
        return d


class FgImageStats(Analyzer):
    """
    Analyzer to extract foreground label properties for each case(image and label).

    Args:
        image_key: the key to find image data in the callable function input (data)
        label_key: the key to find label data in the callable function input (data)

    Examples:

    .. code-block:: python

        import numpy as np
        from monai.auto3dseg import FgImageStats

        input = {}
        input['image'] = np.random.rand(1,30,30,30)
        input['label'] = np.ones([30,30,30])
        analyzer = FgImageStats(image_key='image', label_key='label')
        print(analyzer(input)["image_foreground_stats"])

    """

    def __init__(self, image_key: str, label_key: str, stats_name: str = "image_foreground_stats"):

        self.image_key = image_key
        self.label_key = label_key

        report_format = {ImageStatsKeys.INTENSITY: None}

        super().__init__(stats_name, report_format)
        self.update_ops(ImageStatsKeys.INTENSITY, SampleOperations())

    def __call__(self, data) -> dict:
        """
        Callable to execute the pre-defined functions

        Returns:
            A dictionary. The dict has the key in self.report_format and value
            in a list format. Each element of the value list has stats pre-defined
            by SampleOperations (max, min, ....).

        Raises:
            RuntimeError if the stats report generated is not consistent with the pre-
                defined report_format.

        Note:
            The stats operation uses numpy and torch to compute max, min, and other
            functions. If the input has nan/inf, the stats results will be nan/inf.
        """

        d = dict(data)
        start = time.time()
        restore_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        ndas = [d[self.image_key][i] for i in range(d[self.image_key].shape[0])]
        ndas_label = d[self.label_key]  # (H,W,D)
        nda_foregrounds = [get_foreground_label(nda, ndas_label) for nda in ndas]
        nda_foregrounds = [nda if nda.numel() > 0 else torch.Tensor([0]) for nda in nda_foregrounds]

        # perform calculation
        report = deepcopy(self.get_report_format())

        report[ImageStatsKeys.INTENSITY] = [
            self.ops[ImageStatsKeys.INTENSITY].evaluate(nda_f) for nda_f in nda_foregrounds
        ]

        if not verify_report_format(report, self.get_report_format()):
            raise RuntimeError(f"report generated by {self.__class__} differs from the report format.")

        d[self.stats_name] = report

        torch.set_grad_enabled(restore_grad_state)
        logger.debug(f"Get foreground image stats spent {time.time()-start}")
        return d


class LabelStats(Analyzer):
    """
    Analyzer to extract label stats properties for each case(image and label).

    Args:
        image_key: the key to find image data in the callable function input (data)
        label_key: the key to find label data in the callable function input (data)
        do_ccp: performs connected component analysis. Default is True.

    Examples:

    .. code-block:: python

        import numpy as np
        from monai.auto3dseg import LabelStats

        input = {}
        input['image'] = np.random.rand(1,30,30,30)
        input['label'] = np.ones([30,30,30])
        analyzer = LabelStats(image_key='image', label_key='label')
        print(analyzer(input)["label_stats"])

    """

    def __init__(self, image_key: str, label_key: str, stats_name: str = "label_stats", do_ccp: Optional[bool] = True):

        self.image_key = image_key
        self.label_key = label_key
        self.do_ccp = do_ccp

        report_format: Dict[str, Any] = {
            LabelStatsKeys.LABEL_UID: None,
            LabelStatsKeys.IMAGE_INTST: None,
            LabelStatsKeys.LABEL: [{LabelStatsKeys.PIXEL_PCT: None, LabelStatsKeys.IMAGE_INTST: None}],
        }

        if self.do_ccp:
            report_format[LabelStatsKeys.LABEL][0].update(
                {LabelStatsKeys.LABEL_SHAPE: None, LabelStatsKeys.LABEL_NCOMP: None}
            )

        super().__init__(stats_name, report_format)
        self.update_ops(LabelStatsKeys.IMAGE_INTST, SampleOperations())

        id_seq = ID_SEP_KEY.join([LabelStatsKeys.LABEL, "0", LabelStatsKeys.IMAGE_INTST])
        self.update_ops_nested_label(id_seq, SampleOperations())

    def __call__(self, data):
        """
        Callable to execute the pre-defined functions.

        Returns:
            A dictionary. The dict has the key in self.report_format and value
            in a list format. Each element of the value list has stats pre-defined
            by SampleOperations (max, min, ....).

        Examples:
            output dict contains {
                LabelStatsKeys.LABEL_UID:[0,1,3],
                LabelStatsKeys.IMAGE_INTST: {...},
                LabelStatsKeys.LABEL:[
                    {
                        LabelStatsKeys.PIXEL_PCT: 0.8,
                        LabelStatsKeys.IMAGE_INTST: {...},
                        LabelStatsKeys.LABEL_SHAPE: [...],
                        LabelStatsKeys.LABEL_NCOMP: 1
                    }
                    {
                        LabelStatsKeys.PIXEL_PCT: 0.1,
                        LabelStatsKeys.IMAGE_INTST: {...},
                        LabelStatsKeys.LABEL_SHAPE: [...],
                        LabelStatsKeys.LABEL_NCOMP: 1
                    }
                    {
                        LabelStatsKeys.PIXEL_PCT: 0.1,
                        LabelStatsKeys.IMAGE_INTST: {...},
                        LabelStatsKeys.LABEL_SHAPE: [...],
                        LabelStatsKeys.LABEL_NCOMP: 1
                    }
                ]
                }

        Raises:
            RuntimeError if the stats report generated is not consistent with the pre-
                defined report_format.

        Notes:
            The label class_ID of the dictionary in LabelStatsKeys.LABEL IS NOT the
            index. Instead, the class_ID is the LabelStatsKeys.LABEL_UID with the same
            index. For instance, the last dict in LabelStatsKeys.LABEL in the Examples
            is 3, which is the last element under LabelStatsKeys.LABEL_UID.

            The stats operation uses numpy and torch to compute max, min, and other
            functions. If the input has nan/inf, the stats results will be nan/inf.
        """
        d = dict(data)
        start = time.time()
        if isinstance(d[self.image_key], (torch.Tensor, MetaTensor)) and d[self.image_key].device.type == "cuda":
            using_cuda = True
        else:
            using_cuda = False
        restore_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        ndas = [d[self.image_key][i] for i in range(d[self.image_key].shape[0])]
        ndas_label = d[self.label_key]  # (H,W,D)
        nda_foregrounds = [get_foreground_label(nda, ndas_label) for nda in ndas]
        nda_foregrounds = [nda if nda.numel() > 0 else torch.Tensor([0]) for nda in nda_foregrounds]

        unique_label = unique(ndas_label)
        if isinstance(ndas_label, (MetaTensor, torch.Tensor)):
            unique_label = unique_label.data.cpu().numpy()

        unique_label = unique_label.astype(np.int8).tolist()

        label_substats = []  # each element is one label
        pixel_sum = 0
        pixel_arr = []
        for index in unique_label:
            start_label = time.time()
            label_dict: Dict[str, Any] = {}
            mask_index = ndas_label == index

            nda_masks = [nda[mask_index] for nda in ndas]
            label_dict[LabelStatsKeys.IMAGE_INTST] = [
                self.ops[LabelStatsKeys.IMAGE_INTST].evaluate(nda_m) for nda_m in nda_masks
            ]

            pixel_count = sum(mask_index)
            pixel_arr.append(pixel_count)
            pixel_sum += pixel_count
            if self.do_ccp:  # apply connected component
                if using_cuda:
                    # The back end of get_label_ccp is CuPy
                    # which is unable to automatically release CUDA GPU memory held by PyTorch
                    del nda_masks
                    torch.cuda.empty_cache()
                shape_list, ncomponents = get_label_ccp(mask_index)
                label_dict[LabelStatsKeys.LABEL_SHAPE] = shape_list
                label_dict[LabelStatsKeys.LABEL_NCOMP] = ncomponents

            label_substats.append(label_dict)
            logger.debug(f" label {index} stats takes {time.time() - start_label}")

        for i, _ in enumerate(unique_label):
            label_substats[i].update({LabelStatsKeys.PIXEL_PCT: float(pixel_arr[i] / pixel_sum)})

        report = deepcopy(self.get_report_format())
        report[LabelStatsKeys.LABEL_UID] = unique_label
        report[LabelStatsKeys.IMAGE_INTST] = [
            self.ops[LabelStatsKeys.IMAGE_INTST].evaluate(nda_f) for nda_f in nda_foregrounds
        ]
        report[LabelStatsKeys.LABEL] = label_substats

        if not verify_report_format(report, self.get_report_format()):
            raise RuntimeError(f"report generated by {self.__class__} differs from the report format.")

        d[self.stats_name] = report

        torch.set_grad_enabled(restore_grad_state)
        logger.debug(f"Get label stats spent {time.time()-start}")
        return d


class ImageStatsSumm(Analyzer):
    """
    This summary analyzer processes the values of specific key `stats_name` in a list of dict.
    Typically, the list of dict is the output of case analyzer under the same prefix
    (ImageStats).

    Args:
        stats_name: the key of the to-process value in the dict.
        average: whether to average the statistical value across different image modalities.

    """

    def __init__(self, stats_name: str = "image_stats", average: Optional[bool] = True):
        self.summary_average = average
        report_format = {
            ImageStatsKeys.SHAPE: None,
            ImageStatsKeys.CHANNELS: None,
            ImageStatsKeys.CROPPED_SHAPE: None,
            ImageStatsKeys.SPACING: None,
            ImageStatsKeys.INTENSITY: None,
        }
        super().__init__(stats_name, report_format)

        self.update_ops(ImageStatsKeys.SHAPE, SampleOperations())
        self.update_ops(ImageStatsKeys.CHANNELS, SampleOperations())
        self.update_ops(ImageStatsKeys.CROPPED_SHAPE, SampleOperations())
        self.update_ops(ImageStatsKeys.SPACING, SampleOperations())
        self.update_ops(ImageStatsKeys.INTENSITY, SummaryOperations())

    def __call__(self, data: List[Dict]):
        """
        Callable to execute the pre-defined functions

        Returns:
            A dictionary. The dict has the key in self.report_format and value
            in a list format. Each element of the value list has stats pre-defined
            by SampleOperations (max, min, ....).

        Raises:
            RuntimeError if the stats report generated is not consistent with the pre-
                defined report_format.

        Examples:
            output dict contains a dictionary for all of the following keys{
                ImageStatsKeys.SHAPE:{...}
                ImageStatsKeys.CHANNELS: {...},
                ImageStatsKeys.CROPPED_SHAPE: {...},
                ImageStatsKeys.SPACING: {...},
                ImageStatsKeys.INTENSITY: {...},
                }

        Notes:
            The stats operation uses numpy and torch to compute max, min, and other
            functions. If the input has nan/inf, the stats results will be nan/inf.
        """
        if not isinstance(data, list):
            return ValueError(f"Callable {self.__class__} requires list inputs")

        if len(data) == 0:
            return ValueError(f"Callable {self.__class__} input list is empty")

        if self.stats_name not in data[0]:
            return KeyError(f"{self.stats_name} is not in input data")

        report = deepcopy(self.get_report_format())

        for k in [ImageStatsKeys.SHAPE, ImageStatsKeys.CHANNELS, ImageStatsKeys.CROPPED_SHAPE, ImageStatsKeys.SPACING]:
            v_np = concat_val_to_np(data, [self.stats_name, k])
            report[k] = self.ops[k].evaluate(v_np, dim=(0, 1) if v_np.ndim > 2 and self.summary_average else 0)

        intst_str = ImageStatsKeys.INTENSITY
        op_keys = report[intst_str].keys()  # template, max/min/...
        intst_dict = concat_multikeys_to_dict(data, [self.stats_name, intst_str], op_keys)
        report[intst_str] = self.ops[intst_str].evaluate(intst_dict, dim=None if self.summary_average else 0)

        if not verify_report_format(report, self.get_report_format()):
            raise RuntimeError(f"report generated by {self.__class__} differs from the report format.")

        return report


class FgImageStatsSumm(Analyzer):
    """
    This summary analyzer processes the values of specific key `stats_name` in a list of
    dict. Typically, the list of dict is the output of case analyzer under the similar name
    (FgImageStats).

    Args:
        stats_name: the key of the to-process value in the dict.
        average: whether to average the statistical value across different image modalities.

    """

    def __init__(self, stats_name: str = "image_foreground_stats", average: Optional[bool] = True):
        self.summary_average = average

        report_format = {ImageStatsKeys.INTENSITY: None}
        super().__init__(stats_name, report_format)
        self.update_ops(ImageStatsKeys.INTENSITY, SummaryOperations())

    def __call__(self, data: List[Dict]):
        """
        Callable to execute the pre-defined functions.

        Returns:
            A dictionary. The dict has the key in self.report_format and value
            in a list format. Each element of the value list has stats pre-defined
            by SampleOperations (max, min, ....) and SummaryOperation (max of the
            max, mean of the mean, etc).

        Raises:
            RuntimeError if the stats report generated is not consistent with the pre-
                defined report_format.

        Examples:
            output dict contains a dictionary for all of the following keys{
                ImageStatsKeys.INTENSITY: {...},
                }

        Notes:
            The stats operation uses numpy and torch to compute max, min, and other
            functions. If the input has nan/inf, the stats results will be nan/inf.
        """
        if not isinstance(data, list):
            return ValueError(f"Callable {self.__class__} requires list inputs")

        if len(data) == 0:
            return ValueError(f"Callable {self.__class__} input list is empty")

        if self.stats_name not in data[0]:
            return KeyError(f"{self.stats_name} is not in input data.")

        report = deepcopy(self.get_report_format())
        intst_str = ImageStatsKeys.INTENSITY
        op_keys = report[intst_str].keys()  # template, max/min/...
        intst_dict = concat_multikeys_to_dict(data, [self.stats_name, intst_str], op_keys)

        report[intst_str] = self.ops[intst_str].evaluate(intst_dict, dim=None if self.summary_average else 0)

        if not verify_report_format(report, self.get_report_format()):
            raise RuntimeError(f"report generated by {self.__class__} differs from the report format.")

        return report


class LabelStatsSumm(Analyzer):
    """
    This summary analyzer processes the values of specific key `stats_name` in a list of
    dict. Typically, the list of dict is the output of case analyzer under the similar name
    (LabelStats).

    Args:
        stats_name: the key of the to-process value in the dict.
        average: whether to average the statistical value across different image modalities.

    """

    def __init__(self, stats_name: str = "label_stats", average: Optional[bool] = True, do_ccp: Optional[bool] = True):
        self.summary_average = average
        self.do_ccp = do_ccp

        report_format: Dict[str, Any] = {
            LabelStatsKeys.LABEL_UID: None,
            LabelStatsKeys.IMAGE_INTST: None,
            LabelStatsKeys.LABEL: [{LabelStatsKeys.PIXEL_PCT: None, LabelStatsKeys.IMAGE_INTST: None}],
        }
        if self.do_ccp:
            report_format[LabelStatsKeys.LABEL][0].update(
                {LabelStatsKeys.LABEL_SHAPE: None, LabelStatsKeys.LABEL_NCOMP: None}
            )

        super().__init__(stats_name, report_format)
        self.update_ops(LabelStatsKeys.IMAGE_INTST, SummaryOperations())

        # label-0-'pixel percentage'
        id_seq = ID_SEP_KEY.join([LabelStatsKeys.LABEL, "0", LabelStatsKeys.PIXEL_PCT])
        self.update_ops_nested_label(id_seq, SampleOperations())
        # label-0-'image intensity'
        id_seq = ID_SEP_KEY.join([LabelStatsKeys.LABEL, "0", LabelStatsKeys.IMAGE_INTST])
        self.update_ops_nested_label(id_seq, SummaryOperations())
        # label-0-shape
        id_seq = ID_SEP_KEY.join([LabelStatsKeys.LABEL, "0", LabelStatsKeys.LABEL_SHAPE])
        self.update_ops_nested_label(id_seq, SampleOperations())
        # label-0-ncomponents
        id_seq = ID_SEP_KEY.join([LabelStatsKeys.LABEL, "0", LabelStatsKeys.LABEL_NCOMP])
        self.update_ops_nested_label(id_seq, SampleOperations())

    def __call__(self, data: List[Dict]):
        """
        Callable to execute the pre-defined functions

        Returns:
            A dictionary. The dict has the key in self.report_format and value
            in a list format. Each element of the value list has stats pre-defined
            by SampleOperations (max, min, ....) and SummaryOperation (max of the
            max, mean of the mean, etc).

        Raises:
            RuntimeError if the stats report generated is not consistent with the pre-
                defined report_format.

        Notes:
            The stats operation uses numpy and torch to compute max, min, and other
            functions. If the input has nan/inf, the stats results will be nan/inf.
        """
        if not isinstance(data, list):
            return ValueError(f"Callable {self.__class__} requires list inputs")

        if len(data) == 0:
            return ValueError(f"Callable {self.__class__} input list is empty")

        if self.stats_name not in data[0]:
            return KeyError(f"{self.stats_name} is not in input data")

        report = deepcopy(self.get_report_format())
        # unique class ID
        uid_np = concat_val_to_np(data, [self.stats_name, LabelStatsKeys.LABEL_UID], axis=None, ragged=True)
        unique_label = label_union(uid_np)
        report[LabelStatsKeys.LABEL_UID] = unique_label

        # image intensity
        intst_str = LabelStatsKeys.IMAGE_INTST
        op_keys = report[intst_str].keys()  # template, max/min/...
        intst_dict = concat_multikeys_to_dict(data, [self.stats_name, intst_str], op_keys)
        report[intst_str] = self.ops[intst_str].evaluate(intst_dict, dim=None if self.summary_average else 0)

        detailed_label_list = []
        # iterate through each label
        label_str = LabelStatsKeys.LABEL
        for label_id in unique_label:
            stats = {}

            pct_str = LabelStatsKeys.PIXEL_PCT
            pct_fixed_keys = [self.stats_name, label_str, label_id, pct_str]
            pct_np = concat_val_to_np(data, pct_fixed_keys, allow_missing=True)
            stats[pct_str] = self.ops[label_str][0][pct_str].evaluate(
                pct_np, dim=(0, 1) if pct_np.ndim > 2 and self.summary_average else 0
            )

            if self.do_ccp:
                ncomp_str = LabelStatsKeys.LABEL_NCOMP
                ncomp_fixed_keys = [self.stats_name, LabelStatsKeys.LABEL, label_id, ncomp_str]
                ncomp_np = concat_val_to_np(data, ncomp_fixed_keys, allow_missing=True)
                stats[ncomp_str] = self.ops[label_str][0][ncomp_str].evaluate(
                    ncomp_np, dim=(0, 1) if ncomp_np.ndim > 2 and self.summary_average else 0
                )

                shape_str = LabelStatsKeys.LABEL_SHAPE
                shape_fixed_keys = [self.stats_name, label_str, label_id, LabelStatsKeys.LABEL_SHAPE]
                shape_np = concat_val_to_np(data, shape_fixed_keys, ragged=True, allow_missing=True)
                stats[shape_str] = self.ops[label_str][0][shape_str].evaluate(
                    shape_np, dim=(0, 1) if shape_np.ndim > 2 and self.summary_average else 0
                )
                # label shape is a 3-element value, but the number of labels in each image
                # can vary from 0 to N. So the value in a list format is "ragged"

            intst_str = LabelStatsKeys.IMAGE_INTST
            intst_fixed_keys = [self.stats_name, label_str, label_id, intst_str]
            op_keys = report[label_str][0][intst_str].keys()
            intst_dict = concat_multikeys_to_dict(data, intst_fixed_keys, op_keys, allow_missing=True)
            stats[intst_str] = self.ops[label_str][0][intst_str].evaluate(
                intst_dict, dim=None if self.summary_average else 0
            )

            detailed_label_list.append(stats)

        report[LabelStatsKeys.LABEL] = detailed_label_list

        if not verify_report_format(report, self.get_report_format()):
            raise RuntimeError(f"report generated by {self.__class__} differs from the report format.")

        return report


class FilenameStats(Analyzer):
    """
    This class finds the file path for the loaded image/label and writes the info
    into the data pipeline as a monai transforms.

    Args:
        key: the key to fetch the filename (for example, "image", "label").
        stats_name: the key to store the filename in the output stats report.

    """

    def __init__(self, key: str, stats_name: str) -> None:
        self.key = key
        super().__init__(stats_name, {})

    def __call__(self, data):
        d = dict(data)

        if self.key:  # when there is no (label) file, key can be None
            if self.key not in d:  # check whether image/label is in the data
                raise ValueError(f"Data with key {self.key} is missing.")
            if not isinstance(d[self.key], MetaTensor):
                raise ValueError(f"Value type of {self.key} is not MetaTensor.")
            if ImageMetaKey.FILENAME_OR_OBJ not in d[self.key].meta:
                raise ValueError(f"{ImageMetaKey.FILENAME_OR_OBJ} not found in MetaTensor {d[self.key]}.")
            d[self.stats_name] = d[self.key].meta[ImageMetaKey.FILENAME_OR_OBJ]
        else:
            d[self.stats_name] = "None"

        return d


class ImageHistogram(Analyzer):
    """
    Analyzer to compute intensity histogram.

    Args:
        image_key: the key to find image data in the callable function input (data)
        hist_bins: list of positive integers (one for each channel) for setting the number of bins used to
            compute the histogram. Defaults to [100].
        hist_range: list of lists of two floats (one for each channel) setting the intensity range to
            compute the histogram. Defaults to [-500, 500].

    Examples:

    .. code-block:: python

        import numpy as np
        from monai.auto3dseg.analyzer import ImageHistogram

        input = {}
        input['image'] = np.random.rand(1,30,30,30)
        input['label'] = np.ones([30,30,30])
        analyzer = ImageHistogram(image_key='image')
        print(analyzer(input))

    """

    def __init__(
        self,
        image_key: str,
        stats_name: str = DataStatsKeys.IMAGE_HISTOGRAM,
        hist_bins: Optional[list] = None,
        hist_range: Optional[list] = None,
    ):

        self.image_key = image_key

        # set defaults
        self.hist_bins: list = [100] if hist_bins is None else hist_bins
        self.hist_range: list = [-500, 500] if hist_range is None else hist_range

        report_format = {"counts": None, "bin_edges": None}

        super().__init__(stats_name, report_format)
        self.update_ops(ImageStatsKeys.HISTOGRAM, SampleOperations())

        # check histogram configurations for each channel in list
        if not isinstance(self.hist_bins, list):
            self.hist_bins = [self.hist_bins]
        if not all(isinstance(hr, list) for hr in self.hist_range):
            self.hist_range = [self.hist_range]
        if len(self.hist_bins) != len(self.hist_range):
            raise ValueError(
                f"Number of histogram bins ({len(self.hist_bins)}) and "
                f"histogram ranges ({len(self.hist_range)}) need to be the same!"
            )
        for i, hist_params in enumerate(zip(self.hist_bins, self.hist_range)):
            _hist_bins, _hist_range = hist_params
            if not isinstance(_hist_bins, int) or _hist_bins < 0:
                raise ValueError(f"Expected {i+1}. hist_bins value to be positive integer but got {_hist_bins}")
            if not isinstance(_hist_range, list) or len(_hist_range) != 2:
                raise ValueError(f"Expected {i+1}. hist_range values to be list of length 2 but received {_hist_range}")

    def __call__(self, data) -> dict:
        """
        Callable to execute the pre-defined functions

        Returns:
            A dictionary. The dict has the key in self.report_format and value

        Raises:
            RuntimeError if the stats report generated is not consistent with the pre-
                defined report_format.

        Note:
            The stats operation uses numpy and torch to compute max, min, and other
            functions. If the input has nan/inf, the stats results will be nan/inf.
        """

        d = dict(data)

        ndas = convert_to_numpy(d[self.image_key], wrap_sequence=True)  # (1,H,W,D) or (C,H,W,D)
        nr_channels = np.shape(ndas)[0]

        # adjust histogram params to match channels
        if len(self.hist_bins) == 1:
            self.hist_bins = nr_channels * self.hist_bins
        if len(self.hist_bins) != nr_channels:
            raise ValueError(
                f"There is a mismatch between the number of channels ({nr_channels}) "
                f"and number histogram bins ({len(self.hist_bins)})."
            )
        if len(self.hist_range) == 1:
            self.hist_range = nr_channels * self.hist_range
        if len(self.hist_range) != nr_channels:
            raise ValueError(
                f"There is a mismatch between the number of channels ({nr_channels}) "
                f"and histogram ranges ({len(self.hist_range)})."
            )

        # perform calculation
        reports = []
        for channel in range(nr_channels):
            counts, bin_edges = np.histogram(
                ndas[channel, ...],
                bins=self.hist_bins[channel],
                range=(self.hist_range[channel][0], self.hist_range[channel][1]),
            )
            _report = {"counts": counts.tolist(), "bin_edges": bin_edges.tolist()}
            if not verify_report_format(_report, self.get_report_format()):
                raise RuntimeError(f"report generated by {self.__class__} differs from the report format.")
            reports.append(_report)

        d[self.stats_name] = reports
        return d


class ImageHistogramSumm(Analyzer):
    """
    This summary analyzer processes the values of specific key `stats_name` in a list of dict.
    Typically, the list of dict is the output of case analyzer under the same prefix
    (ImageHistogram).

    Args:
        stats_name: the key of the to-process value in the dict.
        average: whether to average the statistical value across different image modalities.

    """

    def __init__(self, stats_name: str = DataStatsKeys.IMAGE_HISTOGRAM, average: Optional[bool] = True):
        self.summary_average = average
        report_format = {ImageStatsKeys.HISTOGRAM: None}
        super().__init__(stats_name, report_format)

        self.update_ops(ImageStatsKeys.HISTOGRAM, SummaryOperations())

    def __call__(self, data: List[Dict]):
        """
        Callable to execute the pre-defined functions

        Returns:
            A dictionary. The dict has the key in self.report_format and value
            in a list format. Each element of the value list has stats pre-defined
            by SampleOperations (max, min, ....).

        Raises:
            RuntimeError if the stats report generated is not consistent with the pre-
                defined report_format.

        Examples:
            output dict contains a dictionary for all of the following keys{
                ImageStatsKeys.SHAPE:{...}
                ImageStatsKeys.CHANNELS: {...},
                ImageStatsKeys.CROPPED_SHAPE: {...},
                ImageStatsKeys.SPACING: {...},
                ImageStatsKeys.INTENSITY: {...},
                }

        Notes:
            The stats operation uses numpy and torch to compute max, min, and other
            functions. If the input has nan/inf, the stats results will be nan/inf.
        """
        if not isinstance(data, list):
            return ValueError(f"Callable {self.__class__} requires list inputs")

        if len(data) == 0:
            return ValueError(f"Callable {self.__class__} input list is empty")

        if self.stats_name not in data[0]:
            return KeyError(f"{self.stats_name} is not in input data")

        summ_histogram: Dict = {}

        for d in data:
            if not summ_histogram:
                summ_histogram = d[DataStatsKeys.IMAGE_HISTOGRAM]
                # convert to numpy for computing total histogram
                for k in range(len(summ_histogram)):
                    summ_histogram[k]["counts"] = np.array(summ_histogram[k]["counts"])
            else:
                for k in range(len(summ_histogram)):
                    summ_histogram[k]["counts"] += np.array(d[DataStatsKeys.IMAGE_HISTOGRAM][k]["counts"])
                    if np.all(summ_histogram[k]["bin_edges"] != d[DataStatsKeys.IMAGE_HISTOGRAM][k]["bin_edges"]):
                        raise ValueError(
                            f"bin edges are not consistent! {summ_histogram[k]['bin_edges']} vs. "
                            f"{d[DataStatsKeys.IMAGE_HISTOGRAM][k]['bin_edges']}"
                        )

        # convert back to list
        for k in range(len(summ_histogram)):
            summ_histogram[k]["counts"] = summ_histogram[k]["counts"].tolist()

        report = {ImageStatsKeys.HISTOGRAM: summ_histogram}
        if not verify_report_format(report, self.get_report_format()):
            raise RuntimeError(f"report generated by {self.__class__} differs from the report format.")

        return report
