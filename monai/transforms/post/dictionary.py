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
"""
A collection of dictionary-based wrappers around the "vanilla" transforms for model output tensors
defined in :py:class:`monai.transforms.utility.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from monai.transforms.post.array import (
    Activations,
    AsDiscrete,
    KeepLargestConnectedComponent,
    LabelToContour,
    MeanEnsemble,
    VoteEnsemble,
)
from monai.utils import ensure_tuple_rep

__all__ = [
    "Activationsd",
    "AsDiscreted",
    "KeepLargestConnectedComponentd",
    "LabelToContourd",
    "Ensembled",
    "MeanEnsembled",
    "VoteEnsembled",
    "ActivationsD",
    "ActivationsDict",
    "AsDiscreteD",
    "AsDiscreteDict",
    "KeepLargestConnectedComponentD",
    "KeepLargestConnectedComponentDict",
    "LabelToContourD",
    "LabelToContourDict",
    "MeanEnsembleD",
    "MeanEnsembleDict",
    "VoteEnsembleD",
    "VoteEnsembleDict",
]


class Activationsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddActivations`.
    Add activation layers to the input data specified by `keys`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sigmoid: Union[Sequence[bool], bool] = False,
        softmax: Union[Sequence[bool], bool] = False,
        other: Optional[Union[Sequence[Callable], Callable]] = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            sigmoid: whether to execute sigmoid function on model output before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            softmax: whether to execute softmax function on model output before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            other: callable function to execute other activation layers,
                for example: `other = lambda x: torch.tanh(x)`. it also can be a sequence of Callable, each
                element corresponds to a key in ``keys``.

        """
        super().__init__(keys)
        self.sigmoid = ensure_tuple_rep(sigmoid, len(self.keys))
        self.softmax = ensure_tuple_rep(softmax, len(self.keys))
        self.other = ensure_tuple_rep(other, len(self.keys))
        self.converter = Activations()

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.converter(d[key], self.sigmoid[idx], self.softmax[idx], self.other[idx])
        return d


class AsDiscreted(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsDiscrete`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        argmax: Union[Sequence[bool], bool] = False,
        to_onehot: Union[Sequence[bool], bool] = False,
        n_classes: Optional[Union[Sequence[int], int]] = None,
        threshold_values: Union[Sequence[bool], bool] = False,
        logit_thresh: Union[Sequence[float], float] = 0.5,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            argmax: whether to execute argmax function on input data before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            to_onehot: whether to convert input data into the one-hot format. Defaults to False.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            n_classes: the number of classes to convert to One-Hot format. it also can be a
                sequence of int, each element corresponds to a key in ``keys``.
            threshold_values: whether threshold the float value to int number 0 or 1, default is False.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            logit_thresh: the threshold value for thresholding operation, default is 0.5.
                it also can be a sequence of float, each element corresponds to a key in ``keys``.

        """
        super().__init__(keys)
        self.argmax = ensure_tuple_rep(argmax, len(self.keys))
        self.to_onehot = ensure_tuple_rep(to_onehot, len(self.keys))
        self.n_classes = ensure_tuple_rep(n_classes, len(self.keys))
        self.threshold_values = ensure_tuple_rep(threshold_values, len(self.keys))
        self.logit_thresh = ensure_tuple_rep(logit_thresh, len(self.keys))
        self.converter = AsDiscrete()

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.converter(
                d[key],
                self.argmax[idx],
                self.to_onehot[idx],
                self.n_classes[idx],
                self.threshold_values[idx],
                self.logit_thresh[idx],
            )
        return d


class KeepLargestConnectedComponentd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.KeepLargestConnectedComponent`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        applied_labels: Union[Sequence[int], int],
        independent: bool = True,
        connectivity: Optional[int] = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            applied_labels: Labels for applying the connected component on.
                If only one channel. The pixel whose value is not in this list will remain unchanged.
                If the data is in one-hot format, this is the channel indices to apply transform.
            independent: consider several labels as a whole or independent, default is `True`.
                Example use case would be segment label 1 is liver and label 2 is liver tumor, in that case
                you want this "independent" to be specified as False.
            connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. If ``None``, a full
                connectivity of ``input.ndim`` is used.

        """
        super().__init__(keys)
        self.converter = KeepLargestConnectedComponent(applied_labels, independent, connectivity)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class LabelToContourd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LabelToContour`.
    """

    def __init__(self, keys: KeysCollection, kernel_type: str = "Laplace") -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            kernel_type: the method applied to do edge detection, default is "Laplace".

        """
        super().__init__(keys)
        self.converter = LabelToContour(kernel_type=kernel_type)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class Ensembled(MapTransform):
    """
    Base class of dictionary-based ensemble transforms.

    """

    def __init__(
        self,
        keys: KeysCollection,
        ensemble: Callable[[Union[Sequence[torch.Tensor], torch.Tensor]], torch.Tensor],
        output_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be stack and execute ensemble.
                if only 1 key provided, suppose it's a PyTorch Tensor with data stacked on dimension `E`.
            output_key: the key to store ensemble result in the dictionary.
            ensemble: callable method to execute ensemble on specified data.
                if only 1 key provided in `keys`, `output_key` can be None and use `keys` as default.

        Raises:
            TypeError: When ``ensemble`` is not ``callable``.
            ValueError: When ``len(keys) > 1`` and ``output_key=None``. Incompatible values.

        """
        super().__init__(keys)
        if not callable(ensemble):
            raise TypeError(f"ensemble must be callable but is {type(ensemble).__name__}.")
        self.ensemble = ensemble
        if len(self.keys) > 1 and output_key is None:
            raise ValueError("Incompatible values: len(self.keys) > 1 and output_key=None.")
        self.output_key = output_key if output_key is not None else self.keys[0]

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        items: Union[List[torch.Tensor], torch.Tensor]
        if len(self.keys) == 1:
            items = d[self.keys[0]]
        else:
            items = [d[key] for key in self.keys]
        d[self.output_key] = self.ensemble(items)

        return d


class MeanEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MeanEnsemble`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        output_key: Optional[str] = None,
        weights: Optional[Union[Sequence[float], torch.Tensor, np.ndarray]] = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be stack and execute ensemble.
                if only 1 key provided, suppose it's a PyTorch Tensor with data stacked on dimension `E`.
            output_key: the key to store ensemble result in the dictionary.
                if only 1 key provided in `keys`, `output_key` can be None and use `keys` as default.
            weights: can be a list or tuple of numbers for input data with shape: [E, B, C, H, W[, D]].
                or a Numpy ndarray or a PyTorch Tensor data.
                the `weights` will be added to input data from highest dimension, for example:
                1. if the `weights` only has 1 dimension, it will be added to the `E` dimension of input data.
                2. if the `weights` has 3 dimensions, it will be added to `E`, `B` and `C` dimensions.
                it's a typical practice to add weights for different classes:
                to ensemble 3 segmentation model outputs, every output has 4 channels(classes),
                so the input data shape can be: [3, B, 4, H, W, D].
                and add different `weights` for different classes, so the `weights` shape can be: [3, 1, 4].
                for example: `weights = [[[1, 2, 3, 4]], [[4, 3, 2, 1]], [[1, 1, 1, 1]]]`.

        """
        ensemble = MeanEnsemble(weights=weights)
        super().__init__(keys, ensemble, output_key)


class VoteEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.VoteEnsemble`.
    """

    def __init__(
        self, keys: KeysCollection, output_key: Optional[str] = None, num_classes: Optional[int] = None
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be stack and execute ensemble.
                if only 1 key provided, suppose it's a PyTorch Tensor with data stacked on dimension `E`.
            output_key: the key to store ensemble result in the dictionary.
                if only 1 key provided in `keys`, `output_key` can be None and use `keys` as default.
            num_classes: if the input is single channel data instead of One-Hot, we can't get class number
                from channel, need to explicitly specify the number of classes to vote.

        """
        ensemble = VoteEnsemble(num_classes=num_classes)
        super().__init__(keys, ensemble, output_key)


ActivationsD = ActivationsDict = Activationsd
AsDiscreteD = AsDiscreteDict = AsDiscreted
KeepLargestConnectedComponentD = KeepLargestConnectedComponentDict = KeepLargestConnectedComponentd
LabelToContourD = LabelToContourDict = LabelToContourd
MeanEnsembleD = MeanEnsembleDict = MeanEnsembled
VoteEnsembleD = VoteEnsembleDict = VoteEnsembled
