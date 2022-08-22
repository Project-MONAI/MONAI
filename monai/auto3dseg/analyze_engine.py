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
from monai.utils.misc import ImageMetaKey


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
    def __init__(
        self, data: Dict, image_key: str, label_key: str, meta_post_fix: str = "_meta_dict", device: str = "cuda"
    ) -> None:

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

        transform = Compose(list(filter(None, transform_list)))

        image_meta_key = image_key + meta_post_fix
        label_meta_key = label_key + meta_post_fix if label_key else None

        super().__init__(data=transform(data))
        super().update(
            {
                DATA_STATS.BY_CASE_IMAGE_PATH: self.data[image_meta_key][ImageMetaKey.FILENAME_OR_OBJ],
                DATA_STATS.BY_CASE_LABEL_PATH: self.data[label_meta_key][ImageMetaKey.FILENAME_OR_OBJ]
                if label_meta_key
                else "",
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
    def __init__(self, data: Dict, image_key: str, label_key: str, average=True):
        super().__init__(data=data)
        super().update({"image_stats": ImageStatsSummaryAnalyzer("image_stats", average=average)})

        if label_key is not None:
            super().update(
                {
                    "image_foreground_stats": FgImageStatsSummaryAnalyzer("image_foreground_stats", average=average),
                    "label_stats": LabelStatsSummaryAnalyzer("label_stats", average=average),
                }
            )
