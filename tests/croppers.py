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
from copy import deepcopy
from typing import TypeVar, Union

import numpy as np

from monai.data.meta_tensor import MetaTensor
from monai.transforms.croppad.array import CropBase
from monai.transforms.croppad.dictionary import CropBased
from monai.transforms.transform import MapTransform
from tests.utils import TEST_NDARRAYS, assert_allclose


class CropTest(unittest.TestCase):
    Cropper: TypeVar("Cropper", bound=Union[CropBase, CropBased])

    @staticmethod
    def get_arr(shape):
        return np.random.randint(100, size=shape).astype(float)

    def crop_test(self, input_param, input_shape, expected_shape, same_area=None):
        base_comparison = None
        input_image = self.get_arr(input_shape)

        for im_type in TEST_NDARRAYS:
            with self.subTest(im_type=im_type):
                # input parameters, such as roi_start can be numpy, torch, list etc.
                for param_type in TEST_NDARRAYS + (None,):
                    with self.subTest(param_type=param_type):
                        input_param_mod = deepcopy(input_param)
                        if param_type is not None:
                            for k in ("roi_start", "roi_end", "roi_center", "roi_size", "roi_scale"):
                                if k in input_param:
                                    input_param_mod[k] = param_type(input_param[k])
                        im = im_type(input_image)
                        padder = self.Cropper(**input_param_mod)
                        is_map = isinstance(padder, MapTransform)
                        input_data = {"img": im} if is_map else im
                        result = padder(input_data)
                        out_im = result["img"] if is_map else result
                        self.assertTupleEqual(out_im.shape, expected_shape)
                        if same_area is not None:
                            assert_allclose(out_im, im[same_area], type_test=False)
                        # check result is the same regardless of input type
                        if base_comparison is None:
                            base_comparison = out_im
                        else:
                            assert_allclose(out_im, base_comparison)

                        # test inverse
                        inv = padder.inverse(result)
                        inv_im = inv["img"] if is_map else inv
                        self.assertIsInstance(inv_im, MetaTensor)
                        if same_area is not None:
                            assert_allclose(inv_im[same_area], im[same_area], type_test=False)
                        self.assertEqual(inv_im.applied_operations, [])

    def crop_test_value(self, input_param, input_arr, expected_array):
        cropper = self.Cropper(**input_param)
        is_map = isinstance(cropper, MapTransform)
        for im_type in TEST_NDARRAYS:
            with self.subTest(im_type=im_type):
                im = im_type(input_arr)
                input_data = {"img": im} if is_map else im
                result = self.Cropper(**input_param)(input_data)
                out_im = result["img"] if is_map else result
                self.assertIsInstance(out_im, MetaTensor)
                assert_allclose(out_im, expected_array, type_test=False)
