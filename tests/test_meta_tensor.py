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

import random
import string
import unittest
from copy import deepcopy
from typing import Optional, Union

import torch
from parameterized import parameterized

from monai.data.meta_tensor import MetaTensor
from tests.utils import TEST_DEVICES, assert_allclose, skip_if_no_cuda

DTYPES = [[torch.float32], [torch.float64], [torch.float16], [torch.int64], [torch.int32]]
TESTS = []
for _device in TEST_DEVICES:
    for _dtype in DTYPES:
        TESTS.append((*_device, *_dtype))


def rand_string(min_len=5, max_len=10):
    str_size = random.randint(min_len, max_len)
    chars = string.ascii_letters + string.punctuation
    return "".join(random.choice(chars) for _ in range(str_size))


class TestMetaTensor(unittest.TestCase):
    @staticmethod
    def get_im(shape=None, dtype=None, device=None):
        if shape is None:
            shape = shape = (1, 10, 8)
        affine = torch.randint(0, 10, (4, 4))
        meta = {"fname": rand_string()}
        t = torch.rand(shape)
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        m = MetaTensor(t.clone(), affine, meta)
        return m, t

    def check_ids(self, a, b, should_match):
        comp = self.assertEqual if should_match else self.assertNotEqual
        comp(id(a), id(b))

    def check(
        self,
        out: torch.Tensor,
        orig: torch.Tensor,
        *,
        shape: bool = True,
        vals: bool = True,
        ids: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        meta: bool = True,
    ):
        if device is None:
            device = orig.device

        # check the image
        self.assertIsInstance(out, type(orig))
        if shape:
            assert_allclose(torch.as_tensor(out.shape), torch.as_tensor(orig.shape))
        if vals:
            assert_allclose(out, orig)
        self.check_ids(out, orig, ids)
        self.assertTrue(str(device) in str(out.device))

        # check meta and affine are equal and affine is on correct device
        if isinstance(orig, MetaTensor) and isinstance(out, MetaTensor) and meta:
            self.assertEqual(out.meta, orig.meta)
            assert_allclose(out.affine, orig.affine)
            self.assertTrue(str(device) in str(out.affine.device))
            self.check_ids(out.affine, orig.affine, ids)
            self.check_ids(out.meta, orig.meta, ids)

    @parameterized.expand(TESTS)
    def test_as_tensor(self, device, dtype):
        m, t = self.get_im(device=device, dtype=dtype)
        t2 = m.as_tensor()
        self.assertIsInstance(t2, torch.Tensor)
        self.assertNotIsInstance(t2, MetaTensor)
        self.assertIsInstance(m, MetaTensor)
        self.check(t, t2, ids=False)

    @parameterized.expand(TESTS)
    def test_constructor(self, device, dtype):
        m, t = self.get_im(device=device, dtype=dtype)
        m2 = MetaTensor(t.clone(), m.affine, m.meta)
        self.check(m, m2, ids=False, meta=False)

    @parameterized.expand(TESTS)
    @skip_if_no_cuda
    def test_to_cuda(self, device, dtype):
        """Test `to`, `cpu` and `cuda`. For `to`, check args and kwargs."""
        orig, _ = self.get_im(device=device, dtype=dtype)
        m = orig.clone()
        m = m.to("cuda")
        self.check(m, orig, ids=False, device="cuda")
        m = m.cpu()
        self.check(m, orig, ids=False, device="cpu")
        m = m.cuda()
        self.check(m, orig, ids=False, device="cuda")
        m = m.to("cpu")
        self.check(m, orig, ids=False, device="cpu")
        m = m.to(device="cuda")
        self.check(m, orig, ids=False, device="cuda")
        m = m.to(device="cpu")
        self.check(m, orig, ids=False, device="cpu")

    @parameterized.expand(TESTS)
    def test_copy(self, device, dtype):
        m, _ = self.get_im(device=device, dtype=dtype)
        # shallow copy
        a = m
        self.check(a, m, ids=True)
        # deepcopy
        a = deepcopy(m)
        self.check(a, m, ids=False)
        # clone
        a = m.clone()
        self.check(a, m, ids=False)

    @parameterized.expand(TESTS)
    def test_add(self, device, dtype):
        m1, t1 = self.get_im(device=device, dtype=dtype)
        m2, t2 = self.get_im(device=device, dtype=dtype)
        self.check(m1 + m2, t1 + t2, ids=False)
        self.check(torch.add(m1, m2), t1 + t2, ids=False)
        self.check(torch.add(input=m1, other=m2), t1 + t2, ids=False)
        self.check(torch.add(m1, other=m2), t1 + t2, ids=False)
        m3 = deepcopy(m2)
        t3 = deepcopy(m2)
        m3 += 3
        t3 += 3
        self.check(m3, t3, ids=False)
        # check torch.Tensor+MetaTensor and MetaTensor+torch.Tensor
        self.check(torch.add(m1, t2), t1 + t2, ids=False)
        self.check(torch.add(t2, m1), t1 + t2, ids=False)

    @parameterized.expand(TEST_DEVICES)
    def test_conv(self, device):
        im, _ = self.get_im((1, 3, 10, 8, 12), device=device)
        conv = torch.nn.Conv3d(im.shape[1], 5, 3, device=device)
        out = conv(im)
        self.check(out, im, shape=False, vals=False, ids=False)

    @parameterized.expand(TESTS)
    def test_stack(self, device, dtype):
        numel = 3
        ims = [self.get_im(device=device, dtype=dtype)[0] for _ in range(numel)]
        stacked = torch.stack(ims)
        self.assertIsInstance(stacked, MetaTensor)
        assert_allclose(stacked.affine, ims[0].affine)
        self.assertEqual(stacked.meta, ims[0].meta)

    # TODO
    # collate
    # decollate
    # dataset
    # dataloader
    # torchscript
    # matplotlib
    # pickling


if __name__ == "__main__":
    unittest.main()
