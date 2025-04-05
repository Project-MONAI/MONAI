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

from json import dumps
from itertools import product
import time
from typing import Mapping, Sequence
import unittest

import torch
from parameterized import parameterized

from monai.data import MetaTensor, create_test_image_2d, Dataset, ThreadDataLoader, DataLoader
from monai.engines.evaluator import SupervisedEvaluator
from monai.engines.utils import IterationEvents, engine_apply_transform
from monai.transforms import Compose, EnsureChannelFirstd, Resized, Transposed, Invertd, Spacingd
from monai.transforms.utility.array import SimulateDelay
from monai.transforms.utility.dictionary import Lambdad
from monai.utils.misc import first
from monai.utils.enums import CommonKeys

# from tests.test_utils import TEST_DEVICES
TEST_DEVICES = [[torch.device("cpu")]]


class TestInvertDict(unittest.TestCase):

    def setUp(self):
        self.orig_size = (60, 60)
        img, seg = create_test_image_2d(*self.orig_size, 2, 10, num_seg_classes=2)
        self.img = MetaTensor(img, meta={"original_channel_dim": float("nan"), "pixdim": [1.0, 1.0]})
        # self.seg = MetaTensor(seg, meta={"original_channel_dim": float("nan"), "pixdim": [1.0, 1.0]})
        self.key = CommonKeys.IMAGE
        self.new_pixdim = 2.0
        self.new_size = (55, 70)


        def _print(x):
            # print("PRE",f"{id(x):x}",type(x), x.shape, len(x.applied_operations),flush=True)
            # time.sleep(0.01)
            print("PRE",f"{id(x):x}",type(x).__name__, x.shape, len(x.applied_operations),flush=True)
            return x

        self.preprocessing = Compose(
            [
                EnsureChannelFirstd(self.key),
                # Resized(self.key, self.new_size),
                Spacingd(self.key, pixdim=[self.new_pixdim] * 2),
                # Transposed(self.key, (0, 2, 1)),
                Lambdad(self.key, func=_print)
            ]
        )

        
        self.postprocessing = Compose([
            # Lambdad(self.key, func=_print), 
            Invertd(CommonKeys.PRED, transform=self.preprocessing, orig_keys=self.key)
        ])

    # @parameterized.expand(TEST_DEVICES)
    # def test_dataloader_read(self, device):
    #     test_data = [{self.key: self.img.clone().to(device)} for _ in range(4)]
    #     ds = Dataset(test_data, transform=self.preprocessing)
    #     dl = ThreadDataLoader(ds,  num_workers=0, batch_size=2)
    #     # dl = DataLoader(ds,num_workers=0, batch_size=2)

    #     alldata=list(dl)

    # @parameterized.expand(TEST_DEVICES)
    # def test_simple_processing(self, device):
    #     item = {self.key: self.img.to(device)}
    #     pre = self.preprocessing(item)

    #     nw = int(self.orig_size[0] / self.new_pixdim)
    #     nh = int(self.orig_size[1] / self.new_pixdim)

    #     self.assertTupleEqual(pre[self.key].shape, (1, nh, nw))
    #     self.assertTrue(len(pre[self.key].applied_operations) > 0)

    #     post = self.postprocessing(pre)

    #     self.assertTupleEqual(post[self.key].shape, (1, *self.orig_size))



    # @parameterized.expand(product(sum(TEST_DEVICES,[]),[True, False]))
    # def test_dataset_dataloader(self, device,use_threads):
    #     batch_size=2
    #     dl_type=ThreadDataLoader if use_threads else DataLoader

    #     ds = Dataset([{self.key: self.img.to(device)} for _ in range(20)], transform=self.preprocessing)

    #     self.assertGreater(len(ds[0][self.key].applied_operations), 0, "Applied operations are missing")

    #     dl = dl_type(ds,num_workers=0, batch_size=batch_size)

    #     batch=first(dl)

    #     self.assertEqual(len(batch[self.key].applied_operations), batch_size)
    #     self.assertGreater(len(batch[self.key].applied_operations[0]), 0, "Applied operations are missing")

    #     # batch[CommonKeys.PRED] = batch[self.key]
    #     # post_batch=engine_apply_transform(batch=batch,output={},transform=self.postprocessing)



    @parameterized.expand(TEST_DEVICES)
    def test_workflow(self, device):
        test_data = [{self.key: self.img.clone().to(device)} for _ in range(4)]
        batch_size=2
        ds = Dataset(test_data, transform=self.preprocessing)
        dl = ThreadDataLoader(ds,  num_workers=0, batch_size=batch_size)
        # dl = DataLoader(ds,num_workers=0, batch_size=batch_size)

        class AssertAppliedOps(torch.nn.Module):
            def forward(self,x):
                assert len(x.applied_operations)==x.shape[0]
                assert all(len(a)>0 for a in x.applied_operations)
                return x

        # def _print(x):
        #     print(type(x), id(x), x.shape, len(x.applied_operations))
        #     del x.applied_operations[:]
        #     return x

        # postprocessing = Compose([
        #     Lambdad(self.key, func=_print),
        # ])

            

        evaluator = SupervisedEvaluator(
            device=device, 
            network=AssertAppliedOps(), 
            postprocessing=self.postprocessing,
            val_data_loader=dl
        )

        # def tensor_struct_info(tstruct):
        #     if isinstance(tstruct, torch.Tensor):
        #         return f"{id(tstruct):x} {tuple(tstruct.shape)} {tstruct.dtype} {len(getattr(tstruct,"applied_operations",[]))}"
        #     elif isinstance(tstruct, Sequence):
        #         return list(map(tensor_struct_info, tstruct))
        #     elif isinstance(tstruct, Mapping):
        #         return {k: tensor_struct_info(v) for k, v in tstruct.items()}
        #     else:
        #         return repr(tstruct)

        # @evaluator.on(IterationEvents.MODEL_COMPLETED)
        # def _run_postprocessing(engine:SupervisedEvaluator) -> None:
        #     print("\n===================\n")
        #     # print("Batch:",dumps(tensor_struct_info(engine.state.batch),indent=2),flush=True)
        #     print("Output:",dumps(tensor_struct_info(engine.state.output),indent=2),flush=True)

        #     for i, (b, o) in enumerate(zip(engine.state.batch, engine.state.output)):
        #         # print("Post:",dumps(tensor_struct_info(o),indent=2),flush=True)
        #         engine.state.batch[i], engine.state.output[i] = engine_apply_transform(b, o, self.postprocessing)

        # # evaluator._register_postprocessing(self.postprocessing)

        evaluator.run()

        # self.assertTrue(len(evaluator.state.batch[0][self.key].applied_operations)>0)



if __name__ == "__main__":
    unittest.main()
