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

from abc import abstractmethod
from collections.abc import Sequence
from typing import Tuple
from monai.config import KeysCollection
import torch
from monai.transforms import Transform, MapTransform
from monai.utils.misc import ensure_tuple
from math import sqrt, ceil


class Mixer(Transform):
    def __init__(self, batch_size: int, alpha: float = 1.0) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"Expected positive number, but got {alpha = }")
        self._sampler = torch.distributions.beta.Beta(alpha, alpha)
        self.batch_size = batch_size

    def sample_params(self):
        """
        Sometimes you need may to apply the same transform to different tensors.
        The idea is to get a sample and then apply it with apply_mixup() as often
        as needed.
        """
        return self._sampler.sample((self.batch_size,)), torch.randperm(self.batch_size)

    @classmethod
    @abstractmethod
    def apply(cls, params: Tuple[torch.Tensor, torch.Tensor], data: torch.Tensor):
        raise NotImplementedError()


class MixUp(Mixer):
    """MixUp as described in:
    Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz.
    mixup: Beyond Empirical Risk Minimization, ICLR 2018
    """

    def __init__(self, batch_size: int, alpha: float = 1.0) -> None:
        super().__init__(batch_size, alpha)

    @classmethod
    def apply(cls, params: Tuple[torch.Tensor, torch.Tensor], data: torch.Tensor):
        weight, perm = params
        nsamples, *dims = data.shape
        if len(weight) != nsamples:
            raise ValueError(f"Expected batch of size: {len(weight)}, but got {nsamples}")

        if len(dims) not in [3, 4]:
            raise ValueError("Unexpected number of dimensions")

        mixweight = weight[(Ellipsis,) + (None,) * len(dims)]
        return mixweight * data + (1 - mixweight) * data[perm, ...]

    def __call__(self, data: torch.Tensor, labels: torch.Tensor | None = None):
        if labels is None:
            return self.apply(self.sample_params(), data)

        params = self.sample_params()
        return self.apply(params, data), self.apply(params, labels)


class MixUpd(MapTransform):
    """MixUp as described in:
    Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz.
    mixup: Beyond Empirical Risk Minimization, ICLR 2018

    Notice that the mixup transformation will be the same for all entries
    for consistency, i.e. images and labels must be applied the same augmenation.
    """

    def __init__(
        self,
        keys: KeysCollection,
        batch_size: int,
        alpha: float = 1.0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mixup = MixUp(batch_size, alpha)

    def __call__(self, data):
        result = dict(data)
        params = self.mixup.sample_params()
        for k in self.keys:
            result[k] = self.mixup.apply(params, data[k])
        return result


class CutMix(Mixer):
    """CutMix augmentation as described in:
    Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features,
    ICCV 2019
    """

    def __init__(self, batch_size: int, alpha: float = 1.0) -> None:
        super().__init__(batch_size, alpha)

    @classmethod
    def apply(cls, params: Tuple[torch.Tensor, torch.Tensor], data: torch.Tensor):
        weights, perm = params
        nsamples, _, *dims = data.shape
        if len(weights) != nsamples:
            raise ValueError(f"Expected batch of size: {len(weights)}, but got {nsamples}")

        mask = torch.ones_like(data)
        for s, weight in enumerate(weights):
            coords = [torch.randint(0, d, size=(1,)) for d in dims]
            lengths = [d * sqrt(1 - weight) for d in dims]
            idx = [slice(None)] + [slice(c, min(ceil(c + ln), d)) for c, ln, d in zip(coords, lengths, dims)]
            mask[s][idx] = 0

        return mask * data + (1 - mask) * data[perm, ...]

    @classmethod
    def apply_on_labels(cls, params: Tuple[torch.Tensor, torch.Tensor], labels: torch.Tensor):
        return MixUp.apply(params, labels)

    def __call__(self, data: torch.Tensor, labels: torch.Tensor | None = None):
        params = self.sample_params()
        augmented = self.apply(params, data)
        return (augmented, MixUp.apply(params, labels)) if labels is not None else augmented


class CutMixd(MapTransform):
    """CutMix augmentation as described in:
    Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features,
    ICCV 2019

    Notice that the mixture weights will be the same for all entries
    for consistency, i.e. images and labels must be aggregated with the same weights,
    but the random crops are not.
    """

    def __init__(
        self,
        keys: KeysCollection,
        batch_size: int,
        label_keys: KeysCollection | None = None,
        alpha: float = 1.0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mixer = CutMix(batch_size, alpha)
        self.label_keys = ensure_tuple(label_keys) if label_keys is not None else []

    def __call__(self, data):
        result = dict(data)
        params = self.mixer.sample_params()
        for k in self.keys:
            result[k] = self.mixer.apply(params, data[k])
        for k in self.label_keys:
            result[k] = self.mixer.apply_on_labels(params, data[k])
        return result


class CutOut(Mixer):
    """Cutout as described in the paper:
    Terrance DeVries, Graham W. Taylor
    Improved Regularization of Convolutional Neural Networks with Cutout
    arXiv:1708.04552
    """

    @classmethod
    def apply(cls, params: Tuple[torch.Tensor, torch.Tensor], data: torch.Tensor):
        weights, _ = params
        nsamples, _, *dims = data.shape
        if len(weights) != nsamples:
            raise ValueError(f"Expected batch of size: {len(weights)}, but got {nsamples}")

        mask = torch.ones_like(data)
        for s, weight in enumerate(weights):
            coords = [torch.randint(0, d, size=(1,)) for d in dims]
            lengths = [d * sqrt(1 - weight) for d in dims]
            idx = [slice(None)] + [slice(c, min(ceil(c + ln), d)) for c, ln, d in zip(coords, lengths, dims)]
            mask[s][idx] = 0

        return mask * data

    def __call__(self, data: torch.Tensor):
        return self.apply(self.sample_params(), data)


class CutOutd(MapTransform):
    """Cutout as described in the paper:
    Terrance DeVries, Graham W. Taylor
    Improved Regularization of Convolutional Neural Networks with Cutout
    arXiv:1708.04552

    Notice that the cutout is different for every entry in the dictionary.
    """

    def __init__(self, keys: KeysCollection, batch_size: int, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.cutout = CutOut(batch_size)

    def __call__(self, data):
        result = dict(data)
        for k in self.keys:
            result[k] = self.cutout(data[k])
        return result
