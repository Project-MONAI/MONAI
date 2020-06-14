# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from monai.metrics.metric import PartialMetric
from monai.metrics.functional.rocauc import compute_roc_auc


class ROCAUC(PartialMetric):
    """
    Class based API to either compute ROCAUC directly or compute it over
    indivudally added samples
    """

    def __init__(self,
                 to_onehot_y: bool = False,
                 softmax: bool = False,
                 average: Optional[str] = "macro",
                 ):
        """
        Args:
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            softmax: whether to add softmax function to `y_pred` before computation. Defaults to False.
            average (`macro|weighted|micro|None`): type of averaging performed if not binary
                classification. Default is 'macro'.

                - 'macro': calculate metrics for each label, and find their unweighted mean.
                this does not take label imbalance into account.
                - 'weighted': calculate metrics for each label, and find their average,
                weighted by support (the number of true instances for each label).
                - 'micro': calculate metrics globally by considering each element of the label
                indicator matrix as a label.
                - None: the scores for each class are returned.
        """
        super().__init__()
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.average = average
        
        self.y_pred_stored = []
        self.y_stored = []

    def __call__(self,
                 y_pred: torch.Tensor,
                 y: torch.Tensor,
                 ):
        return compute_roc_auc(y_pred, y,
                               to_onehot_y=to_onehot_y,
                               softmax=softmax,
                               average=average,
                               )

    def add_sample(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        self.y_pred_stored.append(y_pred)
        self.y_stored.append(y)

    def evaluate(self) -> float:
        y_pred = torch.cat(self.y_pred_stored)
        y = torch.cat(self.y_stored)
        return self(y_pred, y)

    def reset(self) -> None:
        self.y_pred_stored = []
        self.y_stored = []
