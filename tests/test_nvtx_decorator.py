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
from parameterized import parameterized

from monai.transforms import (
    Compose,
    Flip,
    FlipD,
    RandAdjustContrast,
    RandFlip,
    Randomizable,
    Rotate90,
    ToTensor,
    ToTensorD,
)
from monai.utils import Range, optional_import

_, has_nvtx = optional_import("torch._C._nvtx", descriptor="NVTX is not installed. Are you sure you have a CUDA build?")


TEST_CASE_ARRAY_0 = [
    np.random.randn(3, 3),
]
TEST_CASE_ARRAY_1 = [
    np.random.randn(3, 10, 10),
]

TEST_CASE_DICT_0 = [
    {"image": np.random.randn(3, 3)},
]
TEST_CASE_DICT_1 = [
    {"image": np.random.randn(3, 10, 10)},
]

TEST_CASE_TORCH_0 = [
    torch.randn(3, 3),
]
TEST_CASE_TORCH_1 = [
    torch.randn(3, 10, 10),
]


class TestNVTXRangeDecorator(unittest.TestCase):
    @parameterized.expand([TEST_CASE_ARRAY_0, TEST_CASE_ARRAY_1])
    @unittest.skipUnless(has_nvtx, "CUDA is required for NVTX Range!")
    def test_tranform_array(self, input):
        transforms = Compose(
            [
                Range("random flip")(Flip()),
                Range()(ToTensor()),
            ]
        )
        # Apply transforms
        output = transforms(input)

        # Decorate with NVTX Range
        transforms1 = Range()(transforms)
        transforms2 = Range("Transforms2")(transforms)
        transforms3 = Range(name="Transforms3", methods="__call__")(transforms)

        # Apply transforms with Range
        output1 = transforms1(input)
        output2 = transforms2(input)
        output3 = transforms3(input)

        # Check the outputs
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(output1, torch.Tensor)
        self.assertIsInstance(output2, torch.Tensor)
        self.assertIsInstance(output3, torch.Tensor)
        np.testing.assert_equal(output.numpy(), output1.numpy())
        np.testing.assert_equal(output.numpy(), output1.numpy())
        np.testing.assert_equal(output.numpy(), output3.numpy())

    @parameterized.expand([TEST_CASE_DICT_0, TEST_CASE_DICT_1])
    @unittest.skipUnless(has_nvtx, "CUDA is required for NVTX Range!")
    def test_tranform_dict(self, input):
        transforms = Compose(
            [
                Range("random flip dict")(FlipD(keys="image")),
                Range()(ToTensorD("image")),
            ]
        )
        # Apply transforms
        output = transforms(input)["image"]

        # Decorate with NVTX Range
        transforms1 = Range()(transforms)
        transforms2 = Range("Transforms2")(transforms)
        transforms3 = Range(name="Transforms3", methods="__call__")(transforms)

        # Apply transforms with Range
        output1 = transforms1(input)["image"]
        output2 = transforms2(input)["image"]
        output3 = transforms3(input)["image"]

        # Check the outputs
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(output1, torch.Tensor)
        self.assertIsInstance(output2, torch.Tensor)
        self.assertIsInstance(output3, torch.Tensor)
        np.testing.assert_equal(output.numpy(), output1.numpy())
        np.testing.assert_equal(output.numpy(), output2.numpy())
        np.testing.assert_equal(output.numpy(), output3.numpy())

    @parameterized.expand([TEST_CASE_ARRAY_1])
    @unittest.skipUnless(has_nvtx, "CUDA is required for NVTX Range!")
    def test_tranform_randomized(self, input):
        # Compose deterministic and randomized transforms
        transforms = Compose(
            [
                Range("flip")(Flip()),
                Rotate90(),
                Range()(RandAdjustContrast(prob=0.0)),
                Range("random flip")(RandFlip(prob=1.0)),
                ToTensor(),
            ]
        )
        # Apply transforms
        output = transforms(input)

        # Decorate with NVTX Range
        transforms1 = Range()(transforms)
        transforms2 = Range("Transforms2")(transforms)
        transforms3 = Range(name="Transforms3", methods="__call__")(transforms)

        # Apply transforms with Range
        output1 = transforms1(input)
        output2 = transforms2(input)
        output3 = transforms3(input)

        # Check if the outputs are equal
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(output1, torch.Tensor)
        self.assertIsInstance(output2, torch.Tensor)
        self.assertIsInstance(output3, torch.Tensor)
        np.testing.assert_equal(output.numpy(), output1.numpy())
        np.testing.assert_equal(output.numpy(), output2.numpy())
        np.testing.assert_equal(output.numpy(), output3.numpy())

        # Check if the first randomized is RandAdjustContrast
        for tran in transforms.transforms:
            if isinstance(tran, Randomizable):
                self.assertIsInstance(tran, RandAdjustContrast)
                break

    @parameterized.expand([TEST_CASE_TORCH_0, TEST_CASE_TORCH_1])
    @unittest.skipUnless(has_nvtx, "CUDA is required for NVTX Range!")
    def test_network(self, input):
        # Create a network
        model = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Sigmoid(),
        )

        # Forward
        output = model(input)

        # Decorate with NVTX Range
        model1 = Range()(model)
        model2 = Range("Model2")(model)
        model3 = Range(name="Model3", methods="forward")(model)

        # Forward with Range
        output1 = model1(input)
        output2 = model2(input)
        output3 = model3(input)

        # Check the outputs
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(output1, torch.Tensor)
        self.assertIsInstance(output2, torch.Tensor)
        self.assertIsInstance(output3, torch.Tensor)
        np.testing.assert_equal(output.numpy(), output1.numpy())
        np.testing.assert_equal(output.numpy(), output2.numpy())
        np.testing.assert_equal(output.numpy(), output3.numpy())

    @parameterized.expand([TEST_CASE_TORCH_0, TEST_CASE_TORCH_1])
    @unittest.skipUnless(has_nvtx, "CUDA is required for NVTX Range!")
    def test_loss(self, input):
        # Create a network and loss
        model = torch.nn.Sigmoid()
        loss = torch.nn.BCELoss()
        pred = model(input)
        target = torch.empty_like(input).random_(2)

        # Loss evaluation
        output = loss(pred, target)

        # Decorate with NVTX Range
        loss1 = Range()(loss)
        loss2 = Range("Loss2")(loss)
        loss3 = Range(name="Loss3", methods="forward")(loss)

        # Loss evaluation with Range
        output1 = loss1(pred, target)
        output2 = loss2(pred, target)
        output3 = loss3(pred, target)

        # Check the outputs
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(output1, torch.Tensor)
        self.assertIsInstance(output2, torch.Tensor)
        self.assertIsInstance(output3, torch.Tensor)
        np.testing.assert_equal(output.numpy(), output1.numpy())
        np.testing.assert_equal(output.numpy(), output2.numpy())
        np.testing.assert_equal(output.numpy(), output3.numpy())

    @unittest.skipUnless(has_nvtx, "CUDA is required for NVTX Range!")
    def test_context_manager(self):
        model = torch.nn.Sigmoid()
        loss = torch.nn.BCELoss()

        with Range():
            input = torch.randn(3, requires_grad=True)
            target = torch.empty(3).random_(2)

        with Range("Model"):
            output = loss(model(input), target)
            output.backward()


if __name__ == "__main__":
    unittest.main()
