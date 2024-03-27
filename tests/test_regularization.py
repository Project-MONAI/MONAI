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

import torch

from monai.transforms import CutMix, CutMixd, CutOut, MixUp, MixUpd
from monai.utils import set_determinism


class TestMixup(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(seed=0)

    def tearDown(self) -> None:
        set_determinism(None)

    def test_mixup(self):
        for dims in [2, 3]:
            shape = (6, 3) + (32,) * dims
            sample = torch.rand(*shape, dtype=torch.float32)
            mixup = MixUp(6, 1.0)
            output = mixup(sample)
            self.assertEqual(output.shape, sample.shape)
            self.assertTrue(any(not torch.allclose(sample, mixup(sample)) for _ in range(10)))

        with self.assertRaises(ValueError):
            MixUp(6, -0.5)

        mixup = MixUp(6, 0.5)
        for dims in [2, 3]:
            with self.assertRaises(ValueError):
                shape = (5, 3) + (32,) * dims
                sample = torch.rand(*shape, dtype=torch.float32)
                mixup(sample)

    def test_mixupd(self):
        for dims in [2, 3]:
            shape = (6, 3) + (32,) * dims
            t = torch.rand(*shape, dtype=torch.float32)
            sample = {"a": t, "b": t}
            mixup = MixUpd(["a", "b"], 6)
            output = mixup(sample)
            self.assertTrue(torch.allclose(output["a"], output["b"]))

        with self.assertRaises(ValueError):
            MixUpd(["k1", "k2"], 6, -0.5)


class TestCutMix(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(seed=0)

    def tearDown(self) -> None:
        set_determinism(None)

    def test_cutmix(self):
        for dims in [2, 3]:
            shape = (6, 3) + (32,) * dims
            sample = torch.rand(*shape, dtype=torch.float32)
            cutmix = CutMix(6, 1.0)
            output = cutmix(sample)
            self.assertEqual(output.shape, sample.shape)
            self.assertTrue(any(not torch.allclose(sample, cutmix(sample)) for _ in range(10)))

    def test_cutmixd(self):
        for dims in [2, 3]:
            shape = (6, 3) + (32,) * dims
            t = torch.rand(*shape, dtype=torch.float32)
            label = torch.randint(0, 1, shape)
            sample = {"a": t, "b": t, "lbl1": label, "lbl2": label}
            cutmix = CutMixd(["a", "b"], 6, label_keys=("lbl1", "lbl2"))
            output = cutmix(sample)
            # croppings are different on each application
            self.assertTrue(not torch.allclose(output["a"], output["b"]))
            # but mixing of labels is not affected by it
            self.assertTrue(torch.allclose(output["lbl1"], output["lbl2"]))


class TestCutOut(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(seed=0)

    def tearDown(self) -> None:
        set_determinism(None)

    def test_cutout(self):
        for dims in [2, 3]:
            shape = (6, 3) + (32,) * dims
            sample = torch.rand(*shape, dtype=torch.float32)
            cutout = CutOut(6, 1.0)
            output = cutout(sample)
            self.assertEqual(output.shape, sample.shape)
            self.assertTrue(any(not torch.allclose(sample, cutout(sample)) for _ in range(10)))


if __name__ == "__main__":
    unittest.main()
