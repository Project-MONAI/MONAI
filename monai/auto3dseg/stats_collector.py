import numpy as np
from abc import abstractmethod, ABC
from copy import deepcopy
from collections import UserDict

import torch
from typing import Any, Dict, List, Tuple
from functools import partial
from monai.utils.enums import StrEnum
from monai.transforms.utils_pytorch_numpy_unification import max, mean, median, min, percentile, std

from monai.data.meta_tensor import MetaTensor

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    EnsureTyped,
    Lambdad,
    SqueezeDimd,
    CropForeground,
    ToDeviced,
    ToCupy,
    transform,
)

from monai.utils import min_version, optional_import
from monai.utils.misc import label_union

measure_np, has_measure = optional_import("skimage.measure", "0.14.2", min_version)
cp, has_cp = optional_import("cupy")
cucim, has_cucim = optional_import("cucim")

def get_foreground_image(image: MetaTensor) -> np.ndarray:
    """
    Get a foreground image by removing all-zero rectangles on the edges of the image
    Note for the developer: update select_fn if the foreground is defined differently.

    Args:
        image: ndarray image to segment.

    Returns:
        ndarray of foreground image by removing all-zero edges.

    Notes:
        the size of the ouput is smaller than the input.
    """
    copper = CropForeground(select_fn=lambda x: x > 0)
    image_foreground = copper(image)
    return image_foreground

def get_foreground_label(image: MetaTensor, label: MetaTensor) -> MetaTensor:
    """
    Get foreground image pixel values and mask out the non-labeled area.

    Args
        image: ndarray image to segment.
        label: ndarray the image input and annotated with class IDs.

    Returns:
        1D array of foreground image with label > 0
    """
    label_foreground = MetaTensor(image[label > 0])
    return label_foreground

def get_label_ccp(mask_index: MetaTensor, use_gpu: bool = True) -> Tuple[List[Any], int]:
    """
    Find all connected components and their bounding shape. Backend can be cuPy/cuCIM or Numpy
    depending on the hardware.

    Args:
        mask_index: a binary mask
        use_gpu: a switch to use GPU/CUDA or not. If GPU is unavailable, CPU will be used
            regardless of this setting

    """
    shape_list = []
    if mask_index.device.type == "cuda" and has_cp and has_cucim and use_gpu:
        mask_cupy = ToCupy()(mask_index.short())
        labeled = cucim.skimage.measure.label(mask_cupy)
        vals = cp.unique(labeled[cp.nonzero(labeled)])

        for ncomp in vals:
            comp_idx = cp.argwhere(labeled == ncomp)
            comp_idx_min = cp.min(comp_idx, axis=0).tolist()
            comp_idx_max = cp.max(comp_idx, axis=0).tolist()
            bbox_shape = [comp_idx_max[i] - comp_idx_min[i] + 1 for i in range(len(comp_idx_max))]
            shape_list.append(bbox_shape)
        ncomponents = len(vals)

    elif has_measure:
        labeled, ncomponents = measure_np.label(mask_index.data.cpu().numpy(), background=-1, return_num=True)
        for ncomp in range(1, ncomponents + 1):
            comp_idx = np.argwhere(labeled == ncomp)
            comp_idx_min = np.min(comp_idx, axis=0).tolist()
            comp_idx_max = np.max(comp_idx, axis=0).tolist()
            bbox_shape = [comp_idx_max[i] - comp_idx_min[i] + 1 for i in range(len(comp_idx_max))]
            shape_list.append(bbox_shape)
    else:
        raise RuntimeError("Cannot find one of the following required dependencies: {cuPy+cuCIM} or {scikit-image}")

    return shape_list, ncomponents


class DATA_STATS(StrEnum):
    """
    A set of keys for dataset statistical analysis module

    """
    SUMMARY = "stats_summary"
    BY_CASE = "stats_by_cases"
    BY_CASE_IMAGE_PATH = "image"
    BY_CASE_LABEL_PATH = "label"

class IMAGE_STATS(StrEnum):
    """

    """
    SHAPE = "shape"
    CHANNELS = "channels"
    CROPPED_SHAPE = "cropped_shape"
    SPACING = "spacing"
    INTENSITY = "intensity"

class LABEL_STATS(StrEnum):
    """

    """
    LABEL_UID = "labels"
    PIXEL_PCT = "pixel_percentage"
    IMAGE_INT = "image_intensity"
    LABEL = "label"
    LABEL_SHAPE = "shape"
    LABEL_NCOMP = "ncomponents"

class Operations(UserDict):
    def evaluate(self, data: Any, **kwargs) -> dict:
        return {k: v(data, **kwargs) for k, v in self.data.items() if callable(v)}

class SampleOperations(Operations):
    # todo: missing value/nan/inf
    def __init__(self) -> None:
        self.data = {
            "max": max,
            "mean": mean,
            "median": median,
            "min": min,
            "stdev": std,
            "percentile": partial(percentile, q=[0.5, 10, 90, 99.5])
        }
        self.data_addon = {
            "percentile_00_5": ("percentile", 0),
            "percentile_10_0": ("percentile", 1),
            "percentile_90_0": ("percentile", 2),
            "percentile_99_5": ("percentile", 3),
        }

    def evaluate(self, data: Any, **kwargs) -> dict:
        ret = super().evaluate(data, **kwargs)
        for k, v in self.data_addon.items():
            cache = v[0]
            idx = v[1]
            if isinstance(v, tuple) and cache in ret:
                ret.update({k: ret[cache][idx]})

        return ret

class SummaryOperations(Operations):
    def __init__(self) -> None:
        self.data = {
                "max": max,
                "mean": mean,
                "median": mean,
                "min": min,
                "stdev": mean,
                "percentile_00_5": mean,
                "percentile_10_0": mean,
                "percentile_90_0": mean,
                "percentile_99_5": mean,
            }

    def evaluate(self, data: Any, **kwargs) -> dict:
        return {k: v(data[k], **kwargs) for k, v in self.data.items() if callable(v)}

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
        self.case_analyzer_name = case_analyzer_name
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


    def concat_np(self, key: str, data):
        return np.concatenate([[np.array(d[self.case_analyzer_name][key]) for d in data]])

    def concat_to_dict(self, key: str, ld_data):
        """
        Pinpointing the key in data structure: list of dicts and concat the value
        """
        values = [d[self.case_analyzer_name][key] for d in ld_data]  # ld: list of dicts
        # analysis is a list of list
        key_values = {}
        for k in values[0][0]:
            key_values[k] = np.concatenate([[val[0][k].cpu().numpy() for val in values]]) #gpu/cpu issue

        return key_values

    def __call__(self, data):
        analysis = deepcopy(self.get_report_format())

        axis = 0 # todo: if self.summary_average and data[...].shape > 2, axis = (0, 1)
        for key in [IMAGE_STATS.SHAPE, IMAGE_STATS.CHANNELS, IMAGE_STATS.CROPPED_SHAPE, IMAGE_STATS.SPACING]:
            analysis[key] = self.ops[key].evaluate(self.concat_np(key, data), dim=axis)

        axis = None if self.summary_average else 0
        analysis[IMAGE_STATS.INTENSITY] = self.ops[IMAGE_STATS.INTENSITY].evaluate(
            self.concat_to_dict(IMAGE_STATS.INTENSITY, data), dim=axis)

        return analysis


class FgImageStatsSummaryAnalyzer(Analyzer):
    def __init__(self, case_analyzer_name: str, average=True):
        self.case_analyzer_name = case_analyzer_name
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
        values = [d[self.case_analyzer_name][key] for d in ld_data]  # ld: list of dicts
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
        self.case_analyzer_name = case_analyzer_name
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

    def concat_to_dict(self, key: str, data):
        """
        Pinpointing the key in data structure: list of dicts and concat the value
        """
        values = [d[self.case_analyzer_name][key] for d in data]
        # analysis is a list of list
        key_values = {}
        for k in values[0][0]:
            key_values[k] = np.concatenate([[val[0][k].cpu().numpy() for val in values]]) #gpu/cpu issue

        return key_values

    def concat_label_to_dict(self, label_id: int, key: str, data):

        values = []
        for d in data:
            if label_id in d[self.case_analyzer_name][LABEL_STATS.LABEL_UID]:
                idx = d[self.case_analyzer_name][LABEL_STATS.LABEL_UID].index(label_id)
                values.append(d[self.case_analyzer_name][LABEL_STATS.LABEL][idx][key])

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

        unique_label = label_union(self.concat_np(LABEL_STATS.LABEL_UID, data))
        pixel_summary = self.concat_ldd(LABEL_STATS.PIXEL_PCT, data)

        axis = None if self.summary_average else 0

        analysis[LABEL_STATS.LABEL_UID] = unique_label
        analysis[LABEL_STATS.PIXEL_PCT] = [{k: mean(v)} for k, v in pixel_summary.items()]
        analysis[LABEL_STATS.IMAGE_INT] = self.ops[LABEL_STATS.IMAGE_INT].evaluate(
            self.concat_to_dict(LABEL_STATS.IMAGE_INT, data), dim=axis)

        analysis[LABEL_STATS.LABEL] = []
        for label_id in unique_label:
            stats = {}
            for key in [LABEL_STATS.IMAGE_INT, LABEL_STATS.LABEL_SHAPE, LABEL_STATS.LABEL_NCOMP]:
                stats[key] = self.ops[key].evaluate(self.concat_label_to_dict(label_id, key, data), dim=axis)
            analysis[LABEL_STATS.LABEL].append(stats)

        return analysis


class AnalyzeEngine:
    def __init__(self, data) -> None:
        self.data = data
        self.analyzers = {}

    def update(self, analyzer: Dict[str, callable]):
        self.analyzers.update(analyzer)

    def __call__(self):
        ret = {}
        for k, analyzer in self.analyzers.items():
            if callable(analyzer):
                ret.update({k: analyzer(self.data)})
            elif isinstance(analyzer, str):
                ret.update({k: analyzer})
        return ret

class SegAnalyzeCaseEngine(AnalyzeEngine):
    def __init__(self,
        data: Dict,
        image_key: str,
        label_key: str,
        meta_post_fix: str = "_meta_dict",
        device: str = "cuda",
        ) -> None:

        keys = [image_key] if label_key is None else [image_key, label_key]

        transform_list = [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=keys, device=device, non_blocking=True),
            Orientationd(keys=keys, axcodes="RAS"),
            EnsureTyped(keys=keys, data_type="tensor"),
            Lambdad(
                keys=label_key, func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x
                ) if label_key else None,
            SqueezeDimd(keys=["label"], dim=0) if label_key else None,
        ]

        transform = Compose(list(filter(None, transform_list)))

        image_meta_key = image_key + meta_post_fix
        label_meta_key = label_key + meta_post_fix if label_key else None

        super().__init__(data=transform(data))
        super().update({
            DATA_STATS.BY_CASE_IMAGE_PATH: self.data[image_meta_key]["filename_or_obj"],
            DATA_STATS.BY_CASE_LABEL_PATH: self.data[label_meta_key]["filename_or_obj"] if label_meta_key else "",
            "image_stats": ImageStatsCaseAnalyzer(image_key, label_key),
            "image_foreground_stats": FgImageStatsCasesAnalyzer(image_key, label_key),
            "label_stats": LabelStatsCaseAnalyzer(image_key, label_key),
        })

class SegAnalyzeSummaryEngine(AnalyzeEngine):
    def __init__(self, data: Dict, average=True):
        super().__init__(data=data)
        super().update({
            "image_stats": ImageStatsSummaryAnalyzer("image_stats", average=average),
            "image_foreground_stats": FgImageStatsSummaryAnalyzer("image_foreground_stats", average=average),
            "label_stats": LabelStatsSummaryAnalyzer("label_stats", average=average)
        })
