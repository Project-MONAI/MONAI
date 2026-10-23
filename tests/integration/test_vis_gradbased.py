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
from parameterized import parameterized

from monai.networks.nets import DenseNet, DenseNet121, SEResNet50
from monai.visualize import GuidedBackpropGrad, GuidedBackpropSmoothGrad, SmoothGrad, VanillaGrad


class DenseNetAdjoint(DenseNet121):

    def __call__(self, x, adjoint_info):
        if adjoint_info != 42:
            raise ValueError
        return super().__call__(x)


DENSENET2D = DenseNet121(spatial_dims=2, in_channels=1, out_channels=3)
DENSENET3D = DenseNet(spatial_dims=3, in_channels=1, out_channels=3, init_features=2, growth_rate=2, block_config=(6,))
SENET2D = SEResNet50(spatial_dims=2, in_channels=3, num_classes=4)
SENET3D = SEResNet50(spatial_dims=3, in_channels=3, num_classes=4)
DENSENET2DADJOINT = DenseNetAdjoint(spatial_dims=2, in_channels=1, out_channels=3)

TESTS = []
for type in (VanillaGrad, SmoothGrad, GuidedBackpropGrad, GuidedBackpropSmoothGrad):
    # 2D densenet
    TESTS.append([type, DENSENET2D, (1, 1, 48, 64)])
    # 3D densenet
    TESTS.append([type, DENSENET3D, (1, 1, 6, 6, 6)])
    # 2D senet
    TESTS.append([type, SENET2D, (1, 3, 64, 64)])
    # 3D senet
    TESTS.append([type, SENET3D, (1, 3, 8, 8, 48)])
    # 2D densenet - adjoint
    TESTS.append([type, DENSENET2DADJOINT, (1, 1, 48, 64)])


class TestGradientClassActivationMap(unittest.TestCase):

    @parameterized.expand(TESTS)
    def test_shape(self, vis_type, model, shape):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # optionally test for adjoint info
        kwargs = {"adjoint_info": 42} if isinstance(model, DenseNetAdjoint) else {}

        model.to(device)
        model.eval()
        vis = vis_type(model)
        x = torch.rand(shape, device=device)
        result = vis(x, **kwargs)
        self.assertTupleEqual(result.shape, x.shape)


class TestSmoothGradSampleBatchSize(unittest.TestCase):

    @parameterized.expand([[0], [-1], [1.5], [True], [False], ["2"]])
    def test_invalid_sample_batch_size(self, value):
        with self.assertRaises(ValueError):
            SmoothGrad(DENSENET2D, sample_batch_size=value)


class TestSmoothGradModelState(unittest.TestCase):

    def test_mixed_module_modes_are_restored(self):
        model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=3)
        model.train()
        # a submodule deliberately kept in eval mode must stay there
        frozen = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
        frozen.eval()
        before = [m.training for m in model.modules()]

        vis = SmoothGrad(model, n_samples=2, sample_batch_size=2, verbose=False)
        vis._resolve_index(torch.rand(1, 1, 48, 64), None)

        self.assertEqual([m.training for m in model.modules()], before)

    def test_mode_restored_when_forward_raises(self):
        model = DenseNetAdjoint(spatial_dims=2, in_channels=1, out_channels=3)
        model.train()
        before = [m.training for m in model.modules()]

        vis = SmoothGrad(model, n_samples=2, sample_batch_size=2, verbose=False)
        with self.assertRaises(ValueError):
            vis._resolve_index(torch.rand(1, 1, 48, 64), None, adjoint_info=0)

        self.assertEqual([m.training for m in model.modules()], before)


class TestGradBatchContract(unittest.TestCase):

    @parameterized.expand([[VanillaGrad], [GuidedBackpropGrad]])
    def test_get_grad_rejects_batch(self, vis_type):
        vis = vis_type(DENSENET2D)
        with self.assertRaisesRegex(ValueError, "batch size of 1"):
            vis(torch.rand(2, 1, 48, 64))

    def test_smoothgrad_batched_rejects_input_batch(self):
        vis = SmoothGrad(DENSENET2D, n_samples=2, sample_batch_size=2, verbose=False)
        with self.assertRaisesRegex(ValueError, "input batch size of 1"):
            vis(torch.rand(2, 1, 48, 64))


class TestSmoothGradIndex(unittest.TestCase):

    def test_tensor_index_is_normalised(self):
        vis = SmoothGrad(DENSENET2D, n_samples=2, sample_batch_size=2, verbose=False)
        self.assertEqual(vis._resolve_index(torch.rand(1, 1, 48, 64), torch.tensor([2])), 2)

    def test_multi_element_tensor_index_rejected(self):
        vis = SmoothGrad(DENSENET2D, n_samples=2, sample_batch_size=2, verbose=False)
        with self.assertRaisesRegex(ValueError, "single class index"):
            vis._resolve_index(torch.rand(1, 1, 48, 64), torch.tensor([0, 1]))

    def test_batched_matches_unbatched(self):
        model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=3).eval()
        x = torch.rand(1, 1, 48, 64)

        torch.manual_seed(0)
        expected = SmoothGrad(model, n_samples=4, sample_batch_size=1, verbose=False)(x, index=1)
        torch.manual_seed(0)
        actual = SmoothGrad(model, n_samples=4, sample_batch_size=4, verbose=False)(x, index=1)

        self.assertTupleEqual(actual.shape, x.shape)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
