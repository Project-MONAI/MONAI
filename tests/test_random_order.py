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

from parameterized import parameterized

from monai.data import MetaTensor
from monai.transforms import RandomOrder, TraceableTransform
from monai.transforms.compose import Compose
from monai.utils import set_determinism
from monai.utils.enums import TraceKeys
from tests.test_one_of import A, B, C, Inv, NonInv, X, Y


class InvC(Inv):
    def __init__(self, keys):
        super().__init__(keys)
        self.fwd_fn = lambda x: x + 1
        self.inv_fn = lambda x: x - 1


class InvD(Inv):
    def __init__(self, keys):
        super().__init__(keys)
        self.fwd_fn = lambda x: x * 100
        self.inv_fn = lambda x: x / 100


set_determinism(seed=123)
KEYS = ["x", "y"]
TEST_INVERSES = [
    (RandomOrder((InvC(KEYS), InvD(KEYS))), True, True),
    (Compose((RandomOrder((InvC(KEYS), InvD(KEYS))), RandomOrder((InvD(KEYS), InvC(KEYS))))), True, False),
    (RandomOrder((RandomOrder((InvC(KEYS), InvD(KEYS))), RandomOrder((InvD(KEYS), InvC(KEYS))))), True, False),
    (RandomOrder((Compose((InvC(KEYS), InvD(KEYS))), Compose((InvD(KEYS), InvC(KEYS))))), True, False),
    (RandomOrder((NonInv(KEYS), NonInv(KEYS))), False, False),
]


class TestRandomOrder(unittest.TestCase):
    def test_empty_compose(self):
        c = RandomOrder()
        i = 1
        self.assertEqual(c(i), 1)

    def test_compose_flatten_does_not_affect_random_order(self):
        p = Compose([A(), B(), RandomOrder([C(), Inv(KEYS), Compose([X(), Y()])])])
        f = p.flatten()

        # in this case the flattened transform should be the same.
        def _match(a, b):
            self.assertEqual(type(a), type(b))
            for a_, b_ in zip(a.transforms, b.transforms):
                self.assertEqual(type(a_), type(b_))
                if isinstance(a_, (Compose, RandomOrder)):
                    _match(a_, b_)

        _match(p, f)

    @parameterized.expand(TEST_INVERSES)
    def test_inverse(self, transform, invertible, use_metatensor):
        data = {k: (i + 1) * 10.0 if not use_metatensor else MetaTensor((i + 1) * 10.0) for i, k in enumerate(KEYS)}
        fwd_data1 = transform(data)
        # test call twice won't affect inverse
        fwd_data2 = transform(data)

        if invertible:
            for k in KEYS:
                t = (
                    fwd_data1[TraceableTransform.trace_key(k)][-1]
                    if not use_metatensor
                    else fwd_data1[k].applied_operations[-1]
                )
                # make sure the RandomOrder applied_order was stored
                self.assertEqual(t[TraceKeys.CLASS_NAME], RandomOrder.__name__)

        # call the inverse
        fwd_inv_data1 = transform.inverse(fwd_data1)
        fwd_inv_data2 = transform.inverse(fwd_data2)

        fwd_data = [fwd_data1, fwd_data2]
        fwd_inv_data = [fwd_inv_data1, fwd_inv_data2]
        for i, _fwd_inv_data in enumerate(fwd_inv_data):
            if invertible:
                for k in KEYS:
                    # check transform was removed
                    if not use_metatensor:
                        self.assertTrue(
                            len(_fwd_inv_data[TraceableTransform.trace_key(k)])
                            < len(fwd_data[i][TraceableTransform.trace_key(k)])
                        )
                    # check data is same as original (and different from forward)
                    self.assertEqual(_fwd_inv_data[k], data[k])
                    self.assertNotEqual(_fwd_inv_data[k], fwd_data[i][k])
            else:
                # if not invertible, should not change the data
                self.assertDictEqual(fwd_data[i], _fwd_inv_data)


if __name__ == "__main__":
    unittest.main()
