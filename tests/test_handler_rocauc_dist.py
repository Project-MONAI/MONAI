# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import numpy as np
import torch
import torch.distributed as dist

from monai.handlers import ROCAUC
from tests.utils import DistCall, DistTestCase


class DistributedROCAUC(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2, node_rank=0)
    def test_compute(self):
        auc_metric = ROCAUC(to_onehot_y=True, softmax=True)
        device = f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"
        if dist.get_rank() == 0:
            y_pred = torch.tensor([[0.1, 0.9], [0.3, 1.4]], device=device)
            y = torch.tensor([[0], [1]], device=device)
            auc_metric.update([y_pred, y])

        if dist.get_rank() == 1:
            y_pred = torch.tensor([[0.2, 0.1], [0.1, 0.5]], device=device)
            y = torch.tensor([[0], [1]], device=device)
            auc_metric.update([y_pred, y])

        result = auc_metric.compute()
        np.testing.assert_allclose(0.75, result)


if __name__ == "__main__":
    unittest.main()
