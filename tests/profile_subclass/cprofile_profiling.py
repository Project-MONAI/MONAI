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
Profiling MetaTensor
"""

import cProfile

import torch

from monai.data.meta_tensor import MetaTensor

if __name__ == "__main__":
    n_chan = 3
    for hwd in (10, 200):
        shape = (n_chan, hwd, hwd, hwd)
        a = MetaTensor(torch.rand(shape), meta={"affine": torch.eye(4) * 2, "fname": "something1"})
        b = MetaTensor(torch.rand(shape), meta={"affine": torch.eye(4) * 3, "fname": "something2"})
        cProfile.run("c = a + b", filename=f"out_{hwd}.prof")
