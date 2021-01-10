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
import tensorboard
import torch

from monai.visualize import make_animated_gif_summary


class TestImg2Tensorboard(unittest.TestCase):
    def test_write_gray(self):
        nparr = np.ones(shape=(1, 32, 32, 32), dtype=np.float32)
        summary_object_np = make_animated_gif_summary(
            tag="test_summary_nparr.png",
            image=nparr,
            max_out=1,
            animation_axes=(3,),
            image_axes=(1, 2),
            scale_factor=253.0,
        )
        assert isinstance(
            summary_object_np, tensorboard.compat.proto.summary_pb2.Summary
        ), "make_animated_gif_summary must return a tensorboard.summary object from numpy array"

        tensorarr = torch.tensor(nparr)
        summary_object_tensor = make_animated_gif_summary(
            tag="test_summary_tensorarr.png",
            image=tensorarr,
            max_out=1,
            animation_axes=(3,),
            image_axes=(1, 2),
            scale_factor=253.0,
        )
        assert isinstance(
            summary_object_tensor, tensorboard.compat.proto.summary_pb2.Summary
        ), "make_animated_gif_summary must return a tensorboard.summary object from tensor input"


if __name__ == "__main__":
    unittest.main()
