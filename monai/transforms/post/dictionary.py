# Copyright 2020 MONAI Consortium
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

from typing import Optional

from monai.config.type_definitions import KeysCollection
from monai.utils.misc import ensure_tuple_rep
from monai.transforms.compose import MapTransform
from monai.transforms.post.array import SplitChannel, Activations, AsDiscrete, KeepLargestConnectedComponent


class SplitChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SplitChannel`.
    All the input specified by `keys` should be splitted into same count of data.

    """

    def __init__(self, keys: KeysCollection, output_postfixes, to_onehot=False, num_classes=None):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            output_postfixes (list, tuple): the postfixes to construct keys to store splitted data.
                for example: if the key of input data is `pred` and split 2 classes, the output
                data keys will be: pred_(output_postfixes[0]), pred_(output_postfixes[1])
            to_onehot (bool or list of bool): whether to convert the data to One-Hot format, default is False.
            num_classes (int or list of int): the class number used to convert to One-Hot format
                if `to_onehot` is True.

        """
        super().__init__(keys)
        if not isinstance(output_postfixes, (list, tuple)):
            raise ValueError("must specify key postfixes to store splitted data.")
        self.output_postfixes = output_postfixes
        self.to_onehot = ensure_tuple_rep(to_onehot, len(self.keys))
        self.num_classes = ensure_tuple_rep(num_classes, len(self.keys))
        self.splitter = SplitChannel()

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            rets = self.splitter(d[key], self.to_onehot[idx], self.num_classes[idx])
            assert len(self.output_postfixes) == len(rets), "count of splitted results must match output_postfixes."
            for i, r in enumerate(rets):
                d[f"{key}_{self.output_postfixes[i]}"] = r
        return d


class Activationsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddActivations`.
    Add activation layers to the input data specified by `keys`.
    """

    def __init__(self, keys: KeysCollection, sigmoid=False, softmax=False, other=None):
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            sigmoid (bool, tuple or list of bool): whether to execute sigmoid function on model
                output before transform.
            softmax (bool, tuple or list of bool): whether to execute softmax function on model
                output before transform.
            other (Callable, tuple or list of Callables): callable function to execute other activation layers,
                for example: `other = lambda x: torch.tanh(x)`

        """
        super().__init__(keys)
        self.sigmoid = ensure_tuple_rep(sigmoid, len(self.keys))
        self.softmax = ensure_tuple_rep(softmax, len(self.keys))
        self.other = ensure_tuple_rep(other, len(self.keys))
        self.converter = Activations()

    def __call__(self, data):
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
        argmax: bool = False,
        to_onehot: bool = False,
        n_classes: Optional[int] = None,
        threshold_values: bool = False,
        logit_thresh: float = 0.5,
    ):
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            argmax: whether to execute argmax function on input data before transform.
            to_onehot: whether to convert input data into the one-hot format. Defaults to False.
            n_classes: the number of classes to convert to One-Hot format.
            threshold_values: whether threshold the float value to int number 0 or 1, default is False.
            logit_thresh: the threshold value for thresholding operation, default is 0.5.

        """
        super().__init__(keys)
        self.argmax = ensure_tuple_rep(argmax, len(self.keys))
        self.to_onehot = ensure_tuple_rep(to_onehot, len(self.keys))
        self.n_classes = ensure_tuple_rep(n_classes, len(self.keys))
        self.threshold_values = ensure_tuple_rep(threshold_values, len(self.keys))
        self.logit_thresh = ensure_tuple_rep(logit_thresh, len(self.keys))
        self.converter = AsDiscrete()

    def __call__(self, data):
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
    dictionary-based wrapper of :py:class:monai.transforms.utility.array.KeepLargestConnectedComponent.
    """

    def __init__(
        self,
        keys: KeysCollection,
        applied_labels,
        independent: bool = True,
        connectivity: Optional[int] = None,
        output_postfix: str = "largestcc",
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            applied_labels (int, list or tuple of int): Labels for applying the connected component on.
                If only one channel. The pixel whose value is not in this list will remain unchanged.
                If the data is in one-hot format, this is used to determine what channels to apply.
            independent (bool): consider several labels as a whole or independent, default is `True`.
                Example use case would be segment label 1 is liver and label 2 is liver tumor, in that case
                you want this "independent" to be specified as False.
            connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. If ``None``, a full
                connectivity of ``input.ndim`` is used.

        """
        super().__init__(keys)
        self.converter = KeepLargestConnectedComponent(applied_labels, independent, connectivity)

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.converter(d[key])
        return d


SplitChannelD = SplitChannelDict = SplitChanneld
ActivationsD = ActivationsDict = Activationsd
AsDiscreteD = AsDiscreteDict = AsDiscreted
KeepLargestConnectedComponentD = KeepLargestConnectedComponentDict = KeepLargestConnectedComponentd
