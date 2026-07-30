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
from itertools import product

import torch
from parameterized import parameterized

from monai.data import DataLoader, Dataset, MetaTensor, ThreadDataLoader, create_test_image_2d
from monai.engines.evaluator import SupervisedEvaluator
from monai.transforms import Compose, EnsureChannelFirstd, Invertd, Spacingd
from monai.utils.enums import CommonKeys
from tests.test_utils import TEST_DEVICES, SkipIfNoModule


class TestInvertDict(unittest.TestCase):

    def setUp(self):
        self.orig_size = (60, 60)
        img, _ = create_test_image_2d(*self.orig_size, 2, 10, num_seg_classes=2)
        self.img = MetaTensor(img, meta={"original_channel_dim": float("nan"), "pixdim": [1.0, 1.0]})
        self.key = CommonKeys.IMAGE
        self.pred = CommonKeys.PRED
        self.new_pixdim = 2.0

        self.preprocessing = Compose([EnsureChannelFirstd(self.key), Spacingd(self.key, pixdim=[self.new_pixdim] * 2)])

        self.postprocessing = Compose([Invertd(self.pred, transform=self.preprocessing, orig_keys=self.key)])

    @parameterized.expand(TEST_DEVICES)
    def test_simple_processing(self, device):
        """
        Tests postprocessing operations perform correctly, in particular that `Invertd` does inversion correctly.

        This will apply the preprocessing sequence which resizes the result, then the postprocess sequence which
        returns it to the original shape using Invertd. This tests that the shape of the output is the same as the
        original image. This will also test that Invertd doesn't get confused if transforms in the postprocessing
        sequence are tracing and so adding information to `applied_operations`, this is what `Lambdad` is doing in
        `self.postprocessing`.
        """

        item = {self.key: self.img.to(device)}
        pre = self.preprocessing(item)

        nw = int(self.orig_size[0] / self.new_pixdim)
        nh = int(self.orig_size[1] / self.new_pixdim)

        self.assertTupleEqual(pre[self.key].shape, (1, nh, nw), "Pre-processing did not reshape input correctly")
        self.assertTrue(len(pre[self.key].applied_operations) > 0, "Pre-processing transforms did not trace correctly")

        pre[self.pred] = pre[self.key]  # the inputs are the prediction for this test

        post = self.postprocessing(pre)

        self.assertTupleEqual(
            post[self.pred].shape, (1, *self.orig_size), "Result does not have same shape as original input"
        )

    @parameterized.expand(product(sum(TEST_DEVICES, []), [True, False]))
    @SkipIfNoModule("ignite")
    def test_workflow(self, device, use_threads):
        """
        This tests the interaction between pre and postprocesing transform sequences being executed in parallel.

        When the `ThreadDataLoader` is used to load batches, this is done in parallel at times with the execution of
        the post-process transform sequence. Previously this encountered a race condition at times because the
        `TraceableTransform.tracing` variables of transforms was being toggled in different threads, so at times a
        pre-process transform wouldn't trace correctly and so confuse `Invertd`. Using a `SupervisedEvaluator` is
        the best way to induce this race condition, other methods didn't get the timing right..
        """
        batch_size = 2
        ds_size = 4
        test_data = [{self.key: self.img.clone().to(device)} for _ in range(ds_size)]
        ds = Dataset(test_data, transform=self.preprocessing)
        dl_type = ThreadDataLoader if use_threads else DataLoader
        dl = dl_type(ds, num_workers=0, batch_size=batch_size)

        class AssertAppliedOps(torch.nn.Module):
            def forward(self, x):
                assert len(x.applied_operations) == x.shape[0]
                assert all(len(a) > 0 for a in x.applied_operations)
                return x

        evaluator = SupervisedEvaluator(
            device=device, network=AssertAppliedOps(), postprocessing=self.postprocessing, val_data_loader=dl
        )

        evaluator.run()

        self.assertTupleEqual(evaluator.state.output[0][self.pred].shape, (1, *self.orig_size))


if __name__ == "__main__":
    unittest.main()
