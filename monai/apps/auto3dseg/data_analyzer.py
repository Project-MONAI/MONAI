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

import warnings
from os import path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from monai.apps.utils import get_logger
from monai.auto3dseg import SegSummarizer
from monai.auto3dseg.utils import datafold_read
from monai.bundle import config_parser
from monai.bundle.config_parser import ConfigParser
from monai.data import DataLoader, Dataset
from monai.data.utils import no_collation
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    SqueezeDimd,
    ToDeviced,
)
from monai.utils import StrEnum, min_version, optional_import
from monai.utils.enums import DataStatsKeys, ImageStatsKeys


def strenum_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data.value)


if optional_import("yaml")[1]:
    config_parser.yaml.SafeDumper.add_multi_representer(StrEnum, strenum_representer)

tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")
logger = get_logger(module_name=__name__)

__all__ = ["DataAnalyzer"]


def _argmax_if_multichannel(x):
    return torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x


class DataAnalyzer:
    """
    The DataAnalyzer automatically analyzes given medical image dataset and reports the statistics.
    The module expects file paths to the image data and utilizes the LoadImaged transform to read the
    files, which supports nii, nii.gz, png, jpg, bmp, npz, npy, and dcm formats. Currently, only
    segmentation task is supported, so the user needs to provide paths to the image and label files
    (if have). Also, label data format is preferred to be (1,H,W,D), with the label index in the
    first dimension. If it is in onehot format, it will be converted to the preferred format.

    Args:
        datalist: a Python dictionary storing group, fold, and other information of the medical
            image dataset, or a string to the JSON file storing the dictionary.
        dataroot: user's local directory containing the datasets.
        output_path: path to save the analysis result.
        average: whether to average the statistical value across different image modalities.
        do_ccp: apply the connected component algorithm to process the labels/images
        device: a string specifying hardware (CUDA/CPU) utilized for the operations.
        worker: number of workers to use for parallel processing. If device is cuda/GPU, worker has
            to be 0.
        image_key: a string that user specify for the image. The DataAnalyzer will look it up in the
            datalist to locate the image files of the dataset.
        label_key: a string that user specify for the label. The DataAnalyzer will look it up in the
            datalist to locate the label files of the dataset. If label_key is None, the DataAnalyzer
            will skip looking for labels and all label-related operations.

    Raises:
        ValueError if device is GPU and worker > 0.

    Examples:
        .. code-block:: python

            from monai.apps.auto3dseg.data_analyzer import DataAnalyzer

            datalist = {
                "testing": [{"image": "image_003.nii.gz"}],
                "training": [
                    {"fold": 0, "image": "image_001.nii.gz", "label": "label_001.nii.gz"},
                    {"fold": 0, "image": "image_002.nii.gz", "label": "label_002.nii.gz"},
                    {"fold": 1, "image": "image_001.nii.gz", "label": "label_001.nii.gz"},
                    {"fold": 1, "image": "image_004.nii.gz", "label": "label_004.nii.gz"},
                ],
            }

            dataroot = '/datasets' # the directory where you have the image files (nii.gz)
            DataAnalyzer(datalist, dataroot)

    Notes:
        The module can also be called from the command line interface (CLI).

    For example:

    .. code-block:: bash

        python -m monai.apps.auto3dseg \\
            DataAnalyzer \\
            get_all_case_stats \\
            --datalist="my_datalist.json" \\
            --dataroot="my_dataroot_dir"

    """

    def __init__(
        self,
        datalist: Union[str, Dict],
        dataroot: str = "",
        output_path: str = "./data_stats.yaml",
        average: bool = True,
        do_ccp: bool = True,
        device: Union[str, torch.device] = "cpu",
        worker: int = 2,
        image_key: str = "image",
        label_key: Optional[str] = "label",
    ):
        if path.isfile(output_path):
            warnings.warn(f"File {output_path} already exists and will be overwritten.")
            logger.debug(f"{output_path} will be overwritten by a new datastat.")

        self.datalist = datalist
        self.dataroot = dataroot
        self.output_path = output_path
        self.average = average
        self.do_ccp = do_ccp
        self.device = torch.device(device)
        self.worker = worker
        self.image_key = image_key
        self.label_key = label_key

        if (self.device.type == "cuda") and (worker > 0):
            raise ValueError("CUDA does not support multiple subprocess. If device is GPU, please set worker to 0")

    @staticmethod
    def _check_data_uniformity(keys: List[str], result: Dict):
        """
        Check data uniformity since DataAnalyzer provides no support to multi-modal images with different
        affine matrices/spacings due to monai transforms.

        Args:
            keys: a list of string-type keys under image_stats dictionary.

        Returns:
            False if one of the selected key values is not constant across the dataset images.

        """

        constant_props = [result[DataStatsKeys.SUMMARY][DataStatsKeys.IMAGE_STATS][key] for key in keys]
        for prop in constant_props:
            if "stdev" in prop and np.any(prop["stdev"]):
                return False

        return True

    def get_all_case_stats(self):
        """
        Get all case stats. Caller of the DataAnalyser class. The function iterates datalist and
        call get_case_stats to generate stats. Then get_case_summary is called to combine results.

        Returns:
            A data statistics dictionary containing
                "stats_summary" (summary statistics of the entire datasets). Within stats_summary
                there are "image_stats"  (summarizing info of shape, channel, spacing, and etc
                using operations_summary), "image_foreground_stats" (info of the intensity for the
                non-zero labeled voxels), and "label_stats" (info of the labels, pixel percentage,
                image_intensity, and each individual label in a list)
                "stats_by_cases" (List type value. Each element of the list is statistics of
                an image-label info. Within each element, there are: "image" (value is the
                path to an image), "label" (value is the path to the corresponding label), "image_stats"
                (summarizing info of shape, channel, spacing, and etc using operations),
                "image_foreground_stats" (similar to the previous one but one foreground image), and
                "label_stats" (stats of the individual labels )

        Notes:
            Since the backend of the statistics computation are torch/numpy, nan/inf value
            may be generated and carried over in the computation. In such cases, the output
            dictionary will include .nan/.inf in the statistics.

        """
        summarizer = SegSummarizer(self.image_key, self.label_key, average=self.average, do_ccp=self.do_ccp)
        keys = list(filter(None, [self.image_key, self.label_key]))
        transform_list = [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),  # this creates label to be (1,H,W,D)
            Orientationd(keys=keys, axcodes="RAS"),
            EnsureTyped(keys=keys, data_type="tensor"),
            Lambdad(keys=self.label_key, func=_argmax_if_multichannel) if self.label_key else None,
            SqueezeDimd(keys=["label"], dim=0) if self.label_key else None,
            ToDeviced(keys=keys, device=self.device),
            summarizer,
        ]

        transform = Compose(transforms=list(filter(None, transform_list)))

        files, _ = datafold_read(datalist=self.datalist, basedir=self.dataroot, fold=-1)
        dataset = Dataset(data=files, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.worker, collate_fn=no_collation)
        result = {DataStatsKeys.SUMMARY: {}, DataStatsKeys.BY_CASE: []}
        if not has_tqdm:
            warnings.warn("tqdm is not installed. not displaying the caching progress.")

        for batch_data in tqdm(dataloader) if has_tqdm else dataloader:
            d = batch_data[0]
            stats_by_cases = {
                DataStatsKeys.BY_CASE_IMAGE_PATH: d[DataStatsKeys.BY_CASE_IMAGE_PATH],
                DataStatsKeys.BY_CASE_LABEL_PATH: d[DataStatsKeys.BY_CASE_LABEL_PATH],
                DataStatsKeys.IMAGE_STATS: d[DataStatsKeys.IMAGE_STATS],
            }

            if self.label_key is not None:
                stats_by_cases.update(
                    {
                        DataStatsKeys.FG_IMAGE_STATS: d[DataStatsKeys.FG_IMAGE_STATS],
                        DataStatsKeys.LABEL_STATS: d[DataStatsKeys.LABEL_STATS],
                    }
                )
            result[DataStatsKeys.BY_CASE].append(stats_by_cases)

        result[DataStatsKeys.SUMMARY] = summarizer.summarize(result[DataStatsKeys.BY_CASE])

        if not self._check_data_uniformity([ImageStatsKeys.SPACING], result):
            logger.warning("Data is not completely uniform. MONAI transforms may provide unexpected result")

        ConfigParser.export_config_file(result, self.output_path, fmt="yaml", default_flow_style=None)

        return result
