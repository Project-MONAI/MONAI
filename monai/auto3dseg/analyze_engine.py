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

from typing import Dict, List

from monai.auto3dseg.analyzer import (
    FgImageStatsCaseAnalyzer,
    FgImageStatsSummaryAnalyzer,
    ImageStatsCaseAnalyzer,
    ImageStatsSummaryAnalyzer,
    LabelStatsCaseAnalyzer,
    LabelStatsSummaryAnalyzer,
    FilenameCaseAnalyzer,
)
from monai.transforms import Compose
from monai.utils.enums import DATA_STATS


class SegAnalyzeEngine(Compose):
    """
    SegAnalyzeEngine serializes the operations for data analysis in Auto3Dseg pipeline. It loads
    two types of analyzer functions and execuate differently. The first type of analyzer is
    CaseAnalyzer which is similar to traditional monai transforms. It can be composed with other
    transforms to process the data dict which has image/label keys. The second type of analyzer
    is SummaryAnalyzer which works only on a list of dictionary. Each dictionary is the output
    of the case analyzers on a single dataset.

    Args:
        image_key: a string that user specify for the image. The DataAnalyzer will look it up in the
            datalist to locate the image files of the dataset.
        label_key: a string that user specify for the label. The DataAnalyzer will look it up in the
            datalist to locate the label files of the dataset. If label_key is None, the DataAnalyzer
            will skip looking for labels and all label-related operations.
        do_ccp: apply the connected component algorithm to process the labels/images
    """

    def __init__(self, image_key: str, label_key: str, do_ccp: bool = True) -> None:

        self.image_key = image_key
        self.label_key = label_key

        self.summary_analyzers = []
        super().__init__()

        self.add_analyzer(
            FilenameCaseAnalyzer(image_key, DATA_STATS.BY_CASE_IMAGE_PATH), None
        )
        self.add_analyzer(
            FilenameCaseAnalyzer(label_key, DATA_STATS.BY_CASE_LABEL_PATH), None
        )
        self.add_analyzer(
            ImageStatsCaseAnalyzer(image_key), ImageStatsSummaryAnalyzer()
        )

        if label_key is None:
            return

        self.add_analyzer(
            FgImageStatsCaseAnalyzer(image_key, label_key), FgImageStatsSummaryAnalyzer()
        )

        self.add_analyzer(
            LabelStatsCaseAnalyzer(image_key, label_key, do_ccp=do_ccp), LabelStatsSummaryAnalyzer(do_ccp=do_ccp)
        )

    def add_analyzer(self, case_analyzer, summary_analzyer):
        self.transforms += (case_analyzer, )
        self.summary_analyzers.append(summary_analzyer)

    def summarize(self, data: List[Dict]):
        if not isinstance(data, list):
            raise ValueError(f"{self.__class__} summarize function needs input to be a list of dict")

        report = {}
        if len(data) == 0:
            return report

        if not isinstance(data[0], dict):
            raise ValueError(f"{self.__class__} summarize function needs a list of dict. Now we have {type(data[0])}")

        for analyzer in self.summary_analyzers:
            if callable(analyzer):
                report.update({analyzer.stats_name: analyzer(data)})

        return report
