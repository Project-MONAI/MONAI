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

from monai.utils.misc import ensure_tuple_rep
from monai.transforms.compose import MapTransform
from monai.transforms.post.array import SplitChannel, AddActivations, AsDiscrete


class SplitChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SplitChannel`.
    All the input specified by `keys` should be splitted into same count of data.

    """

    def __init__(self, keys, output_postfixes, to_onehot=False, num_classes=None):
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


class AddActivationsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddActivations`.
    Add activation layers to the input data specified by `keys`.

    """

    def __init__(self, keys, output_postfix="act", add_sigmoid=False, add_softmax=False, other=None):
        """
        Args:
            keys (hashable items): keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            output_postfix (str): the postfix string to construct keys to store converted data.
                for example: if the keys of input data is `pred` and `label`, output_postfix is `act`,
                the output data keys will be: `pred_act`, `label_act`.
            add_sigmoid (bool, tuple or list of bool): whether to add sigmoid function to model
                output before transform.
            add_softmax (bool, tuple or list of bool): whether to add softmax function to model
                output before transform.
            other (Callable, tuple or list of Callables): callable function to execute other activation layers,
                for example: `other = lambda x: torch.tanh(x)`

        """
        super().__init__(keys)
        if not isinstance(output_postfix, str):
            raise ValueError("output_postfix must be a string.")
        self.output_postfix = output_postfix
        self.add_sigmoid = ensure_tuple_rep(add_sigmoid, len(self.keys))
        self.add_softmax = ensure_tuple_rep(add_softmax, len(self.keys))
        self.other = ensure_tuple_rep(other, len(self.keys))
        self.converter = AddActivations()

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            ret = self.converter(d[key], self.add_sigmoid[idx], self.add_softmax[idx], self.other[idx])
            d[f"{key}_{self.output_postfix}"] = ret
        return d


class AsDiscreted(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsDiscrete`.

    """

    def __init__(
        self,
        keys,
        output_postfix="discreted",
        add_argmax=False,
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
            add_argmax (bool): whether to add argmax function to input data before transform.
            to_onehot (bool): whether to convert input data into the one-hot format. Defaults to False.
            n_classes (bool): the number of classes to convert to One-Hot format.
            threshold_values (bool): whether threshold the float value to int number 0 or 1, default is False.
            logit_thresh (float): the threshold value for thresholding operation, default is 0.5.

        """
        super().__init__(keys)
        if not isinstance(output_postfix, str):
            raise ValueError("output_postfix must be a string.")
        self.output_postfix = output_postfix
        self.add_argmax = ensure_tuple_rep(add_argmax, len(self.keys))
        self.to_onehot = ensure_tuple_rep(to_onehot, len(self.keys))
        self.n_classes = ensure_tuple_rep(n_classes, len(self.keys))
        self.threshold_values = ensure_tuple_rep(threshold_values, len(self.keys))
        self.logit_thresh = ensure_tuple_rep(logit_thresh, len(self.keys))
        self.converter = AsDiscrete()

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[f"{key}_{self.output_postfix}"] = self.converter(
                d[key],
                self.add_argmax[idx],
                self.to_onehot[idx],
                self.n_classes[idx],
                self.threshold_values[idx],
                self.logit_thresh[idx],
            )
        return d


AddActivationsD = AddActivationsDict = AddActivationsd
AsDiscreteD = AsDiscreteDict = AsDiscreted
SplitChannelD = SplitChannelDict = SplitChanneld
