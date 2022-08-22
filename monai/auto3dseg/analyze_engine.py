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

from typing import Dict

import torch

from monai.auto3dseg.analyzer import (
    FgImageStatsCasesAnalyzer,
    FgImageStatsSummaryAnalyzer,
    ImageStatsCaseAnalyzer,
    ImageStatsSummaryAnalyzer,
    LabelStatsCaseAnalyzer,
    LabelStatsSummaryAnalyzer,
)
from monai.auto3dseg.utils import get_filename
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
from monai.utils.enums import DATA_STATS


class AnalyzeEngine:
    """
    AnalyzeEngine is a base class to serialize the operations for data analysis in Auto3Dseg pipeline.

    Args:
        data: a dict-type data to run analysis on

    Examples:

    .. code-block:: python

        import numpy as np
        from monai.auto3dseg.analyze_engine import AnalyzeEngine

        engine = AnalyzeEngine()
        engine.update({"max": np.max})
        engine.update({"min": np.min})

        input = np.array([1,2,3,4])
        print(engine(input))

    """

    def __init__(self) -> None:
        self.analyzers = {}
        self.transform = None

    def update(self, analyzer: Dict[str, callable]):
        self.analyzers.update(analyzer)

    def __call__(self, data):
        if self.transform:
            data = self.transform(data)

        ret = {}
        for k, analyzer in self.analyzers.items():
            if callable(analyzer):
                ret.update({k: analyzer(data)})
            elif isinstance(analyzer, str):
                ret.update({k: analyzer})
        return ret


class SegAnalyzeCaseEngine(AnalyzeEngine):
    def __init__(self, image_key: str, label_key: str, meta_post_fix: str = "_meta_dict", device: str = "cuda") -> None:

        super().__init__()
        keys = [image_key] if label_key is None else [image_key, label_key]

        transform_list = [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=keys, device=device, non_blocking=True),
            Orientationd(keys=keys, axcodes="RAS"),
            EnsureTyped(keys=keys, data_type="tensor"),
            Lambdad(keys=label_key, func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x)
            if label_key
            else None,
            SqueezeDimd(keys=["label"], dim=0) if label_key else None,
        ]

        self.transform = Compose(list(filter(None, transform_list)))

        image_meta_key = image_key + meta_post_fix
        label_meta_key = label_key + meta_post_fix if label_key else None

        super().update(
            {
                DATA_STATS.BY_CASE_IMAGE_PATH: lambda data: get_filename(data, meta_key=image_meta_key),
                DATA_STATS.BY_CASE_LABEL_PATH: lambda data: get_filename(data, meta_key=label_meta_key),
                "image_stats": ImageStatsCaseAnalyzer(image_key),
            }
        )

        if label_key is not None:
            super().update(
                {
                    "image_foreground_stats": FgImageStatsCasesAnalyzer(image_key, label_key),
                    "label_stats": LabelStatsCaseAnalyzer(image_key, label_key),
                }
            )


class SegAnalyzeSummaryEngine(AnalyzeEngine):
    def __init__(self, image_key: str, label_key: str, average=True):
        super().__init__()
        super().update({"image_stats": ImageStatsSummaryAnalyzer("image_stats", average=average)})

        if label_key is not None:
            super().update(
                {
                    "image_foreground_stats": FgImageStatsSummaryAnalyzer("image_foreground_stats", average=average),
                    "label_stats": LabelStatsSummaryAnalyzer("label_stats", average=average),
                }
            )
