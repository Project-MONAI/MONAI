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


import numpy as np
import torch
import torch.distributed as dist

from monai.handlers import ConfusionMatrix


def main():
    for compute_sample in [True, False]:
        dist.init_process_group(backend="nccl", init_method="env://")

        torch.cuda.set_device(dist.get_rank())
        metric = ConfusionMatrix(include_background=True, metric_name="tpr", compute_sample=compute_sample)

        if dist.get_rank() == 0:
            y_pred = torch.tensor(
                [
                    [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]],
                ],
                device=torch.device("cuda:0"),
            )
            y = torch.tensor(
                [
                    [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]],
                ],
                device=torch.device("cuda:0"),
            )
            metric.update([y_pred, y])

        if dist.get_rank() == 1:
            y_pred = torch.tensor(
                [[[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 0.0]]]],
                device=torch.device("cuda:1"),
            )
            y = torch.tensor(
                [[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]],
                device=torch.device("cuda:1"),
            )
            metric.update([y_pred, y])

        avg_metric = metric.compute()
        if compute_sample is False:
            avg_metric = avg_metric.item()
            np.testing.assert_allclose(avg_metric, 0.7, rtol=1e-04, atol=1e-04)
        else:
            np.testing.assert_allclose(avg_metric, 0.8333, rtol=1e-04, atol=1e-04)

        dist.destroy_process_group()


# suppose to execute on 2 rank processes
# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
#        --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
#        --master_addr="192.168.1.1" --master_port=1234
#        test_handler_confusion_matrix_dist.py

if __name__ == "__main__":
    main()
