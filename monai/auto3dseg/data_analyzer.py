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
from typing import Dict, Union

import torch

from monai import data
from monai.apps.utils import get_logger
from monai.auto3dseg.utils import datafold_read
from monai.data.utils import no_collation
from monai.utils import min_version, optional_import

tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")
logger = get_logger(module_name=__name__)

__all__ = ["DataAnalyzer"]


from monai.auto3dseg.analyze_engine import DATA_STATS, SegAnalyzeCaseEngine, SegAnalyzeSummaryEngine


class DataAnalyzer:
    """
    The DataAnalyzer automatically analyzes given medical image dataset and reports the statistics.
    The module expects file paths to the image data and utilizes the LoadImaged transform to read the files.
    which supports nii, nii.gz, png, jpg, bmp, npz, npy, and dcm formats. Currently, only segmentation
    problem is supported, so the user needs to provide paths to the image and label files. Also, label
    data format is preferred to be (1,H,W,D), with the label index in the first dimension. If it is in
    onehot format, it will be converted to the preferred format.

    Args:
        datalist: a Python dictionary storing group, fold, and other information of the medical
            image dataset, or a string to the JSON file storing the dictionary.
        dataroot: user's local directory containing the datasets.
        output_path: path to save the analysis result.
        average: whether to average the statistical value across different image modalities.
        do_ccp: apply the connected component algorithm to process the labels/images
        device: a string specifying hardware (CUDA/CPU) utilized for the operations.
        worker: number of workers to use for parallel processing.
        image_key: a string that user specify for the image. The DataAnalyzer will look it up in the
            datalist to locate the image files of the dataset.
        label_key: a string that user specify for the label. The DataAnalyzer will look it up in the
            datalist to locate the label files of the dataset. If label_key is None, the DataAnalyzer
            will skip looking for labels and all label-related operations.

    For example:

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

        python -m monai.apps.auto3dseg \
            DataAnalyzer \
            get_all_case_stats \
            --datalist="my_datalist.json" \
            --dataroot="my_dataroot_dir"

    """

    def __init__(
        self,
        datalist: Union[str, Dict],
        dataroot: str = "",
        output_path: str = "./data_stats.yaml",
        do_ccp: bool = True,
        device: Union[str, torch.device] = "cuda",
        worker: int = 2,
        image_key: str = "image",
        label_key: str = "label",
    ):
        """
        The initializer will load the data and register the functions for data statistics gathering.
        """
        if path.isfile(output_path):
            warnings.warn(f"File {output_path} already exists and will be overwritten.")
            logger.debug(f"{output_path} will be overwritten by a new datastat.")

        self.image_key = image_key
        self.label_key = label_key

        self.output_path = output_path
        self.IMAGE_ONLY = True if label_key is None else False

        self.datalist = datalist
        self.dataroot = dataroot
        self.device = device
        self.worker = worker

    def get_all_case_stats(self):

        keys = list(filter(None, [self.image_key, self.label_key]))
        files, _ = datafold_read(datalist=self.datalist, basedir=self.dataroot, fold=-1)
        ds = data.Dataset(data=files)
        self.dataset = data.DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=self.worker, collate_fn=no_collation
        )

        result = {DATA_STATS.SUMMARY: {}, DATA_STATS.BY_CASE: []}

        case_engine = SegAnalyzeCaseEngine(self.image_key, self.label_key, device=self.device)
        for batch_data in self.dataset:
            result[DATA_STATS.BY_CASE].append(case_engine(batch_data[0]))

        summary_engine = SegAnalyzeSummaryEngine(self.image_key, self.label_key)
        result[DATA_STATS.SUMMARY] = summary_engine(result[DATA_STATS.BY_CASE])

        return result
