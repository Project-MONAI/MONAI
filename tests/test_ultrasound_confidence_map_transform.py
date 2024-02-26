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

from __future__ import annotations

import unittest

import numpy as np
import torch

from monai.transforms import UltrasoundConfidenceMapTransform
from tests.utils import assert_allclose

TEST_INPUT = np.array(
    [
        [1, 2, 3, 23, 13, 22, 5, 1, 2, 3],
        [1, 2, 3, 12, 4, 6, 9, 1, 2, 3],
        [1, 2, 3, 8, 7, 10, 11, 1, 2, 3],
        [1, 2, 3, 14, 15, 16, 17, 1, 2, 3],
        [1, 2, 3, 18, 19, 20, 21, 1, 2, 3],
        [1, 2, 3, 24, 25, 26, 27, 1, 2, 3],
        [1, 2, 3, 28, 29, 30, 31, 1, 2, 3],
        [1, 2, 3, 32, 33, 34, 35, 1, 2, 3],
        [1, 2, 3, 36, 37, 38, 39, 1, 2, 3],
        [1, 2, 3, 40, 41, 42, 43, 1, 2, 3],
    ]
)

TEST_MASK = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

SINK_ALL_OUTPUT = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [
            0.97514489,
            0.96762971,
            0.96164186,
            0.95463443,
            0.9941512,
            0.99023054,
            0.98559401,
            0.98230057,
            0.96601224,
            0.95119599,
        ],
        [
            0.92960533,
            0.92638451,
            0.9056675,
            0.9487176,
            0.9546961,
            0.96165853,
            0.96172303,
            0.92686401,
            0.92122613,
            0.89957239,
        ],
        [
            0.86490963,
            0.85723665,
            0.83798141,
            0.90816201,
            0.90816097,
            0.90815301,
            0.9081427,
            0.85933627,
            0.85146935,
            0.82948586,
        ],
        [
            0.77430346,
            0.76731372,
            0.74372311,
            0.89128774,
            0.89126885,
            0.89125066,
            0.89123521,
            0.76858589,
            0.76106647,
            0.73807776,
        ],
        [
            0.66098109,
            0.65327697,
            0.63090644,
            0.33086588,
            0.3308383,
            0.33081937,
            0.33080718,
            0.6557468,
            0.64825099,
            0.62593375,
        ],
        [
            0.52526945,
            0.51832586,
            0.49709412,
            0.25985059,
            0.25981009,
            0.25977729,
            0.25975222,
            0.52118958,
            0.51426328,
            0.49323164,
        ],
        [
            0.3697845,
            0.36318971,
            0.34424661,
            0.17386804,
            0.17382046,
            0.17377993,
            0.17374668,
            0.36689317,
            0.36036096,
            0.3415582,
        ],
        [
            0.19546374,
            0.1909659,
            0.17319999,
            0.08423318,
            0.08417993,
            0.08413242,
            0.08409104,
            0.19393909,
            0.18947485,
            0.17185031,
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)

SINK_MID_OUTPUT = np.array(
    [
        [
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
        ],
        [
            9.99996103e-01,
            9.99994823e-01,
            9.99993550e-01,
            9.99930863e-01,
            9.99990782e-01,
            9.99984683e-01,
            9.99979000e-01,
            9.99997804e-01,
            9.99995985e-01,
            9.99994325e-01,
        ],
        [
            9.99989344e-01,
            9.99988600e-01,
            9.99984099e-01,
            9.99930123e-01,
            9.99926598e-01,
            9.99824297e-01,
            9.99815032e-01,
            9.99991228e-01,
            9.99990881e-01,
            9.99988462e-01,
        ],
        [
            9.99980787e-01,
            9.99979264e-01,
            9.99975828e-01,
            9.59669286e-01,
            9.59664779e-01,
            9.59656566e-01,
            9.59648332e-01,
            9.99983882e-01,
            9.99983038e-01,
            9.99980732e-01,
        ],
        [
            9.99970181e-01,
            9.99969032e-01,
            9.99965730e-01,
            9.45197806e-01,
            9.45179593e-01,
            9.45163629e-01,
            9.45151458e-01,
            9.99973352e-01,
            9.99973254e-01,
            9.99971098e-01,
        ],
        [
            9.99958608e-01,
            9.99957307e-01,
            9.99953444e-01,
            4.24743523e-01,
            4.24713305e-01,
            4.24694646e-01,
            4.24685271e-01,
            9.99960948e-01,
            9.99961829e-01,
            9.99960347e-01,
        ],
        [
            9.99946675e-01,
            9.99945139e-01,
            9.99940312e-01,
            3.51353224e-01,
            3.51304003e-01,
            3.51268260e-01,
            3.51245366e-01,
            9.99947688e-01,
            9.99950165e-01,
            9.99949512e-01,
        ],
        [
            9.99935877e-01,
            9.99934088e-01,
            9.99928982e-01,
            2.51197134e-01,
            2.51130273e-01,
            2.51080014e-01,
            2.51045852e-01,
            9.99936187e-01,
            9.99939716e-01,
            9.99940022e-01,
        ],
        [
            9.99927846e-01,
            9.99925911e-01,
            9.99920188e-01,
            1.31550973e-01,
            1.31462736e-01,
            1.31394558e-01,
            1.31346069e-01,
            9.99927275e-01,
            9.99932142e-01,
            9.99933313e-01,
        ],
        [
            9.99924204e-01,
            9.99922004e-01,
            9.99915767e-01,
            3.04861147e-04,
            1.95998056e-04,
            0.00000000e00,
            2.05182682e-05,
            9.99923115e-01,
            9.99928835e-01,
            9.99930535e-01,
        ],
    ]
)

SINK_MIN_OUTPUT = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [
            0.99997545,
            0.99996582,
            0.99995245,
            0.99856594,
            0.99898314,
            0.99777223,
            0.99394423,
            0.98588113,
            0.97283215,
            0.96096504,
        ],
        [
            0.99993872,
            0.99993034,
            0.9998832,
            0.9986147,
            0.99848741,
            0.9972981,
            0.99723719,
            0.94157173,
            0.9369832,
            0.91964243,
        ],
        [
            0.99990802,
            0.99989475,
            0.99986873,
            0.98610197,
            0.98610047,
            0.98609749,
            0.98609423,
            0.88741275,
            0.88112911,
            0.86349156,
        ],
        [
            0.99988924,
            0.99988509,
            0.99988698,
            0.98234089,
            0.98233591,
            0.98233065,
            0.98232562,
            0.81475172,
            0.80865978,
            0.79033138,
        ],
        [
            0.99988418,
            0.99988484,
            0.99988323,
            0.86796555,
            0.86795874,
            0.86795283,
            0.86794756,
            0.72418193,
            0.71847704,
            0.70022037,
        ],
        [
            0.99988241,
            0.99988184,
            0.99988103,
            0.85528225,
            0.85527303,
            0.85526389,
            0.85525499,
            0.61716519,
            0.61026209,
            0.59503671,
        ],
        [
            0.99988015,
            0.99987985,
            0.99987875,
            0.84258114,
            0.84257121,
            0.84256042,
            0.84254897,
            0.48997924,
            0.49083978,
            0.46891561,
        ],
        [
            0.99987865,
            0.99987827,
            0.9998772,
            0.83279589,
            0.83278624,
            0.83277384,
            0.83275897,
            0.36345545,
            0.33690244,
            0.35696828,
        ],
        [
            0.99987796,
            0.99987756,
            0.99987643,
            0.82873223,
            0.82872648,
            0.82871803,
            0.82870711,
            0.0,
            0.26106012,
            0.29978657,
        ],
    ]
)

SINK_MASK_OUTPUT = np.array(
    [
        [
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            2.86416400e-01,
            7.93271181e-01,
            5.81341234e-01,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            1.98395623e-01,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            2.66733297e-01,
            2.80741490e-01,
            4.14078784e-02,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            7.91676486e-04,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            1.86244537e-04,
            1.53413401e-04,
            7.85806495e-05,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            5.09797387e-06,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            9.62904581e-07,
            7.23946225e-07,
            3.68824440e-07,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            4.79525316e-08,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            1.50939343e-10,
            1.17724874e-10,
            6.21760843e-11,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            6.08922784e-10,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            2.57593754e-13,
            1.94066716e-13,
            9.83784370e-14,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            9.80828665e-12,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            4.22323494e-16,
            3.17556633e-16,
            1.60789400e-16,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            1.90789819e-13,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            7.72677888e-19,
            5.83029424e-19,
            2.95946659e-19,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            4.97038275e-15,
        ],
        [
            2.71345908e-24,
            5.92006757e-24,
            2.25580089e-23,
            3.82601970e-18,
            3.82835349e-18,
            3.83302158e-18,
            3.84002606e-18,
            8.40760586e-16,
            1.83433696e-15,
            1.11629633e-15,
        ],
    ]
)


class TestUltrasoundConfidenceMapTransform(unittest.TestCase):

    def setUp(self):
        self.input_img_np = np.expand_dims(TEST_INPUT, axis=0)  # mock image (numpy array)
        self.input_mask_np = np.expand_dims(TEST_MASK, axis=0)  # mock mask (numpy array)

        self.input_img_torch = torch.from_numpy(TEST_INPUT).unsqueeze(0)  # mock image (torch tensor)
        self.input_mask_torch = torch.from_numpy(TEST_MASK).unsqueeze(0)  # mock mask (torch tensor)

    def test_parameters(self):
        # Unknown mode
        with self.assertRaises(ValueError):
            UltrasoundConfidenceMapTransform(mode="unknown")

        # Unknown sink_mode
        with self.assertRaises(ValueError):
            UltrasoundConfidenceMapTransform(sink_mode="unknown")

    def test_rgb(self):
        # RGB image
        input_img_rgb = np.expand_dims(np.repeat(self.input_img_np, 3, axis=0), axis=0)
        input_img_rgb_torch = torch.from_numpy(input_img_rgb)

        transform = UltrasoundConfidenceMapTransform(sink_mode="all")
        result_torch = transform(input_img_rgb_torch)
        self.assertIsInstance(result_torch, torch.Tensor)
        assert_allclose(result_torch, torch.tensor(SINK_ALL_OUTPUT), rtol=1e-4, atol=1e-4)
        result_np = transform(input_img_rgb)
        self.assertIsInstance(result_np, np.ndarray)
        assert_allclose(result_np, SINK_ALL_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(sink_mode="mid")
        result_torch = transform(input_img_rgb_torch)
        self.assertIsInstance(result_torch, torch.Tensor)
        assert_allclose(result_torch, torch.tensor(SINK_MID_OUTPUT), rtol=1e-4, atol=1e-4)
        result_np = transform(input_img_rgb)
        self.assertIsInstance(result_np, np.ndarray)
        assert_allclose(result_np, SINK_MID_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(sink_mode="min")
        result_torch = transform(input_img_rgb_torch)
        self.assertIsInstance(result_torch, torch.Tensor)
        assert_allclose(result_torch, torch.tensor(SINK_MIN_OUTPUT), rtol=1e-4, atol=1e-4)
        result_np = transform(input_img_rgb)
        self.assertIsInstance(result_np, np.ndarray)
        assert_allclose(result_np, SINK_MIN_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(sink_mode="mask")
        result_torch = transform(input_img_rgb_torch, self.input_mask_torch)
        self.assertIsInstance(result_torch, torch.Tensor)
        assert_allclose(result_torch, torch.tensor(SINK_MASK_OUTPUT), rtol=1e-4, atol=1e-4)
        result_np = transform(input_img_rgb, self.input_mask_np)
        self.assertIsInstance(result_np, np.ndarray)
        assert_allclose(result_np, SINK_MASK_OUTPUT, rtol=1e-4, atol=1e-4)

    def test_multi_channel_2d(self):
        # 2D multi-channel image
        input_img_rgb = np.expand_dims(np.repeat(self.input_img_np, 17, axis=0), axis=0)
        input_img_rgb_torch = torch.from_numpy(input_img_rgb)

        transform = UltrasoundConfidenceMapTransform(sink_mode="all")
        result_torch = transform(input_img_rgb_torch)
        self.assertIsInstance(result_torch, torch.Tensor)
        assert_allclose(result_torch, torch.tensor(SINK_ALL_OUTPUT), rtol=1e-4, atol=1e-4)
        result_np = transform(input_img_rgb)
        self.assertIsInstance(result_np, np.ndarray)
        assert_allclose(result_np, SINK_ALL_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(sink_mode="mid")
        result_torch = transform(input_img_rgb_torch)
        self.assertIsInstance(result_torch, torch.Tensor)
        assert_allclose(result_torch, torch.tensor(SINK_MID_OUTPUT), rtol=1e-4, atol=1e-4)
        result_np = transform(input_img_rgb)
        self.assertIsInstance(result_np, np.ndarray)
        assert_allclose(result_np, SINK_MID_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(sink_mode="min")
        result_torch = transform(input_img_rgb_torch)
        self.assertIsInstance(result_torch, torch.Tensor)
        assert_allclose(result_torch, torch.tensor(SINK_MIN_OUTPUT), rtol=1e-4, atol=1e-4)
        result_np = transform(input_img_rgb)
        self.assertIsInstance(result_np, np.ndarray)
        assert_allclose(result_np, SINK_MIN_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(sink_mode="mask")
        result_torch = transform(input_img_rgb_torch, self.input_mask_torch)
        self.assertIsInstance(result_torch, torch.Tensor)
        assert_allclose(result_torch, torch.tensor(SINK_MASK_OUTPUT), rtol=1e-4, atol=1e-4)
        result_np = transform(input_img_rgb, self.input_mask_np)
        self.assertIsInstance(result_np, np.ndarray)
        assert_allclose(result_np, SINK_MASK_OUTPUT, rtol=1e-4, atol=1e-4)

    def test_non_one_first_dim(self):
        # Image without first dimension as 1
        input_img_rgb = np.repeat(self.input_img_np, 3, axis=0)
        input_img_rgb_torch = torch.from_numpy(input_img_rgb)

        transform = UltrasoundConfidenceMapTransform(sink_mode="all")
        with self.assertRaises(ValueError):
            transform(input_img_rgb_torch)
        with self.assertRaises(ValueError):
            transform(input_img_rgb)

        transform = UltrasoundConfidenceMapTransform(sink_mode="mid")
        with self.assertRaises(ValueError):
            transform(input_img_rgb_torch)
        with self.assertRaises(ValueError):
            transform(input_img_rgb)

        transform = UltrasoundConfidenceMapTransform(sink_mode="min")
        with self.assertRaises(ValueError):
            transform(input_img_rgb_torch)
        with self.assertRaises(ValueError):
            transform(input_img_rgb)

        transform = UltrasoundConfidenceMapTransform(sink_mode="mask")
        with self.assertRaises(ValueError):
            transform(input_img_rgb_torch, self.input_mask_torch)
        with self.assertRaises(ValueError):
            transform(input_img_rgb, self.input_mask_np)

    def test_no_first_dim(self):
        # Image without first dimension
        input_img_rgb = self.input_img_np[0]
        input_img_rgb_torch = torch.from_numpy(input_img_rgb)

        transform = UltrasoundConfidenceMapTransform(sink_mode="all")
        with self.assertRaises(ValueError):
            transform(input_img_rgb_torch)
        with self.assertRaises(ValueError):
            transform(input_img_rgb)

        transform = UltrasoundConfidenceMapTransform(sink_mode="mid")
        with self.assertRaises(ValueError):
            transform(input_img_rgb_torch)
        with self.assertRaises(ValueError):
            transform(input_img_rgb)

        transform = UltrasoundConfidenceMapTransform(sink_mode="min")
        with self.assertRaises(ValueError):
            transform(input_img_rgb_torch)
        with self.assertRaises(ValueError):
            transform(input_img_rgb)

        transform = UltrasoundConfidenceMapTransform(sink_mode="mask")
        with self.assertRaises(ValueError):
            transform(input_img_rgb_torch, self.input_mask_torch)
        with self.assertRaises(ValueError):
            transform(input_img_rgb, self.input_mask_np)

    def test_sink_all(self):
        transform = UltrasoundConfidenceMapTransform(sink_mode="all")

        # This should not raise an exception for torch tensor
        result_torch = transform(self.input_img_torch)
        self.assertIsInstance(result_torch, torch.Tensor)

        # This should not raise an exception for numpy array
        result_np = transform(self.input_img_np)
        self.assertIsInstance(result_np, np.ndarray)

    def test_sink_mid(self):
        transform = UltrasoundConfidenceMapTransform(sink_mode="mid")

        # This should not raise an exception for torch tensor
        result_torch = transform(self.input_img_torch)
        self.assertIsInstance(result_torch, torch.Tensor)

        # This should not raise an exception for numpy array
        result_np = transform(self.input_img_np)
        self.assertIsInstance(result_np, np.ndarray)

    def test_sink_min(self):
        transform = UltrasoundConfidenceMapTransform(sink_mode="min")

        # This should not raise an exception for torch tensor
        result_torch = transform(self.input_img_torch)
        self.assertIsInstance(result_torch, torch.Tensor)

        # This should not raise an exception for numpy array
        result_np = transform(self.input_img_np)
        self.assertIsInstance(result_np, np.ndarray)

    def test_sink_mask(self):
        transform = UltrasoundConfidenceMapTransform(sink_mode="mask")

        # This should not raise an exception for torch tensor with mask
        result_torch = transform(self.input_img_torch, self.input_mask_torch)
        self.assertIsInstance(result_torch, torch.Tensor)

        # This should not raise an exception for numpy array with mask
        result_np = transform(self.input_img_np, self.input_mask_np)
        self.assertIsInstance(result_np, np.ndarray)

        # This should raise an exception for torch tensor without mask
        with self.assertRaises(ValueError):
            transform(self.input_img_torch)

        # This should raise an exception for numpy array without mask
        with self.assertRaises(ValueError):
            transform(self.input_img_np)

    def test_func(self):
        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="all")
        output = transform(self.input_img_np)
        assert_allclose(output, SINK_ALL_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="mid")
        output = transform(self.input_img_np)
        assert_allclose(output, SINK_MID_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="min")
        output = transform(self.input_img_np)
        assert_allclose(output, SINK_MIN_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="mask")
        output = transform(self.input_img_np, self.input_mask_np)
        assert_allclose(output, SINK_MASK_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="all")
        output = transform(self.input_img_torch)
        assert_allclose(output, torch.tensor(SINK_ALL_OUTPUT), rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="mid")
        output = transform(self.input_img_torch)
        assert_allclose(output, torch.tensor(SINK_MID_OUTPUT), rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="min")
        output = transform(self.input_img_torch)
        assert_allclose(output, torch.tensor(SINK_MIN_OUTPUT), rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="mask")
        output = transform(self.input_img_torch, self.input_mask_torch)
        assert_allclose(output, torch.tensor(SINK_MASK_OUTPUT), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
