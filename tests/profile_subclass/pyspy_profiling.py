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

"""
To be used with py-spy, comparing torch.Tensor, SubTensor, SubWithTorchFunc, MetaTensor
Adapted from https://github.com/pytorch/pytorch/tree/v1.11.0/benchmarks/overrides_benchmark
"""
import argparse

import torch
from min_classes import SubTensor, SubWithTorchFunc  # noqa: F401

from monai.data import MetaTensor  # noqa: F401

Tensor = torch.Tensor

NUM_REPEATS = 1000000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the torch.add for a given class a given number of times.")
    parser.add_argument("tensor_class", metavar="TensorClass", type=str, help="The class to benchmark.")
    parser.add_argument("--nreps", "-n", type=int, default=NUM_REPEATS, help="The number of repeats.")
    args = parser.parse_args()

    TensorClass = globals()[args.tensor_class]
    NUM_REPEATS = args.nreps

    t1 = TensorClass(1)
    t2 = TensorClass(2)

    for _ in range(NUM_REPEATS):
        torch.add(t1, t2)
