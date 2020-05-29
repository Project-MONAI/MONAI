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

from typing import Optional, Hashable

from monai.utils.misc import ensure_tuple_rep
from monai.transforms.compose import MapTransform
from monai.transforms.post.array import SplitChannel, Activations, AsDiscrete, KeepLargestConnectedComponent


class SplitChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SplitChannel`.
    All the input specified by `keys` should be splitted into same count of data.

    """

    def __init__(self, keys: Hashable, output_postfixes, to_onehot=False, num_classes=None):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
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

    def __init__(self, keys: Hashable, output_postfix: str = "act", sigmoid=False, softmax=False, other=None):
        """
        Args:
            keys (hashable items): keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            output_postfix (str): the postfix string to construct keys to store converted data.
                for example: if the keys of input data is `pred` and `label`, output_postfix is `act`,
                the output data keys will be: `pred_act`, `label_act`.
                if set to None, will replace the original data with the same key.
            sigmoid (bool, tuple or list of bool): whether to execute sigmoid function on model
                output before transform.
            softmax (bool, tuple or list of bool): whether to execute softmax function on model
                output before transform.
            other (Callable, tuple or list of Callables): callable function to execute other activation layers,
                for example: `other = lambda x: torch.tanh(x)`
        """
        super().__init__(keys)
        if output_postfix is not None and not isinstance(output_postfix, str):
            raise ValueError("output_postfix must be a string.")
        self.output_postfix = output_postfix
        self.sigmoid = ensure_tuple_rep(sigmoid, len(self.keys))
        self.softmax = ensure_tuple_rep(softmax, len(self.keys))
        self.other = ensure_tuple_rep(other, len(self.keys))
        self.converter = Activations()

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            ret = self.converter(d[key], self.sigmoid[idx], self.softmax[idx], self.other[idx])
            output_key = key if self.output_postfix is None else f"{key}_{self.output_postfix}"
            d[output_key] = ret
        return d


class AsDiscreted(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsDiscrete`.
    """

    def __init__(
        self,
        keys: Hashable,
        output_postfix: str = "discreted",
        argmax=False,
        to_onehot=False,
        n_classes=None,
        threshold_values=False,
        logit_thresh=0.5,
    ):
        """
        Args:
            keys (hashable items): keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            output_postfix (str): the postfix string to construct keys to store converted data.
                for example: if the keys of input data is `pred` and `label`, output_postfix is `discreted`,
                the output data keys will be: `pred_discreted`, `label_discreted`.
                if set to None, will replace the original data with the same key.
            argmax (bool): whether to execute argmax function on input data before transform.
            to_onehot (bool): whether to convert input data into the one-hot format. Defaults to False.
            n_classes (bool): the number of classes to convert to One-Hot format.
            threshold_values (bool): whether threshold the float value to int number 0 or 1, default is False.
            logit_thresh (float): the threshold value for thresholding operation, default is 0.5.
        """
        super().__init__(keys)
        if output_postfix is not None and not isinstance(output_postfix, str):
            raise ValueError("output_postfix must be a string.")
        self.output_postfix = output_postfix
        self.argmax = ensure_tuple_rep(argmax, len(self.keys))
        self.to_onehot = ensure_tuple_rep(to_onehot, len(self.keys))
        self.n_classes = ensure_tuple_rep(n_classes, len(self.keys))
        self.threshold_values = ensure_tuple_rep(threshold_values, len(self.keys))
        self.logit_thresh = ensure_tuple_rep(logit_thresh, len(self.keys))
        self.converter = AsDiscrete()

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            output_key = key if self.output_postfix is None else f"{key}_{self.output_postfix}"
            d[output_key] = self.converter(
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
        keys: Hashable,
        applied_values,
        independent: bool = True,
        background: int = 0,
        connectivity: Optional[int] = None,
        output_postfix: str = "largestcc",
    ):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            applied_values (list or tuple of int): number list for applying the connected component on.
                The pixel whose value is not in this list will remain unchanged.
            independent (bool): consider several labels as a whole or independent, default is `True`.
                Example use case would be segment label 1 is liver and label 2 is liver tumor, in that case
                you want this "independent" to be specified as False.
            background (int): Background pixel value. The over-segmented pixels will be set as this value.
            connectivity (int): Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. If ``None``, a full
                connectivity of ``input.ndim`` is used.
            output_postfix (str): the postfix string to construct keys to store converted data.
                for example: if the keys of input data is `label`, output_postfix is `largestcc`,
                the output data keys will be: `label_largestcc`.
                if set to None, will replace the original data with the same key.
        """
        super().__init__(keys)
        if output_postfix is not None and not isinstance(output_postfix, str):
            raise ValueError("output_postfix must be a string.")
        self.output_postfix = output_postfix
        self.converter = KeepLargestConnectedComponent(applied_values, independent, background, connectivity)

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            output_key = key if self.output_postfix is None else f"{key}_{self.output_postfix}"
            d[output_key] = self.converter(d[key])
        return d


SplitChannelD = SplitChannelDict = SplitChanneld
ActivationsD = ActivationsDict = Activationsd
AsDiscreteD = AsDiscreteDict = AsDiscreted
KeepLargestConnectedComponentD = KeepLargestConnectedComponentDict = KeepLargestConnectedComponentd
