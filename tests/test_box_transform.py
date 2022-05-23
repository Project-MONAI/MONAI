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

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.apps.detection.transforms.dictionary import ConvertBoxModed, ZoomBoxd
from monai.transforms import Invertd, Compose
from tests.utils import TEST_NDARRAYS, assert_allclose
from monai.transforms import CastToTyped

TESTS = []

boxes = [[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 3, 3], [0, 1, 1, 2, 3, 4]]
image_size = [1, 4, 6, 4]
image = np.zeros(image_size)

for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {"box_keys": "boxes", "dst_mode": "xyzwhd"},
            {"boxes": p(boxes), "image": p(image)},
            p([[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]]),
            p([[0, 0, 0, 0, 0, 0], [0, 3, 0, 1, 9, 4.5], [0, 3, 1.5, 1, 9, 6]]),
            p([[1, -6, -1, 1, -6, -1], [1, -3, -1, 2, 3, 3.5], [1, -3, 0.5, 2, 3, 5]]),
            p([[0, 1, 0, 2, 3, 3], [0, 1, 1, 2, 3, 4]]),
            p([[0, 1, 1, 2, 3, 4], [0, 1, 0, 2, 3, 3]]),
        ]
    )


class TestBoxTransform(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, keys, data, 
        expected_convert_result, 
        expected_zoom_result, expected_zoom_keepsize_result, 
        expected_clip_result, expected_flip_result
    ):
        test_dtype = [torch.float16, torch.float32]
        for dtype in test_dtype:
            data = CastToTyped(keys=["image","boxes"],dtype=dtype)(data)
            # test ConvertBoxToStandardModed
            transform_convert_mode = ConvertBoxModed(**keys)
            convert_result = transform_convert_mode(data)
            assert_allclose(convert_result["boxes"], expected_convert_result, type_test=True, device_test=True, atol=0.0)

            invert_transform_convert_mode = Invertd(keys=["boxes"], transform=transform_convert_mode, orig_keys=["boxes"])
            data_back = invert_transform_convert_mode(convert_result)
            assert_allclose(data_back["boxes"], data["boxes"], type_test=False, device_test=False, atol=0.0)

            # test ZoomBoxd
            transform_zoom = ZoomBoxd(
                image_keys = "image",
                box_keys= "boxes",
                box_ref_image_keys="image",
                zoom=[0.5, 3, 1.5],
                keep_size = False
            )
            zoom_result = transform_zoom(data)
            assert_allclose(zoom_result["boxes"], expected_zoom_result, type_test=True, device_test=True, atol=0.0)

            invert_transform_zoom = Invertd(keys=["image","boxes"], transform=transform_zoom, orig_keys=["image","boxes"])
            data_back = invert_transform_zoom(zoom_result)
            assert_allclose(data_back["boxes"],data["boxes"], type_test=False, device_test=False, atol=0.0)
            assert_allclose(data_back["image"],data["image"], type_test=False, device_test=False, atol=0.0)

            transform_zoom = ZoomBoxd(
                image_keys = "image",
                box_keys= "boxes",
                box_ref_image_keys="image",
                zoom=[0.5, 3, 1.5],
                keep_size = True
            )
            zoom_result = transform_zoom(data)
            assert_allclose(zoom_result["boxes"], expected_zoom_keepsize_result, type_test=True, device_test=True, atol=0.0)


if __name__ == "__main__":
    unittest.main()
