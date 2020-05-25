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
from monai.transforms.post.array import SplitChannel, ConvertForMetrics


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


class ConvertForMetricsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertForMetrics`.
    The input specified by `keys` should contains only 2 items(model output, label) or 1 item(model output).

    """
    def __init__(
        self,
        keys,
        output_postfixes,
        add_sigmoid=False,
        add_softmax=False,
        add_argmax=False,
        to_onehot_y_pred=False,
        to_onehot_y=False,
        n_classes=None,
        round_values=False,
        logit_thresh=0.5
    ):
        """
        Args:
            keys (hashable items): keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            output_postfixes (list, tuple): the postfixes to construct keys to store converted data.
                for example: if the keys of input data is `pred` and `label`, the output data keys will be:
                pred_(output_postfixes[0]), label_(output_postfixes[1])
            add_sigmoid (bool): whether to add sigmoid function to y_pred before transform.
            add_softmax (bool): whether to add softmax function to y_pred before transform.
            add_argmax (bool): whether to add argmax function to y_pred before transform.
            to_onehot_y_pred (bool): whether to convert `y_pred` into the one-hot format. Defaults to False.
            to_onehot_y (bool): whether to convert `y` into the one-hot format. Defaults to False.
            n_classes (bool): the number of classes to convert to One-Hot format, if None, use `y_pred.shape[1]`
            round_values (bool): whether round the value to 0 and 1, default is False.
            logit_thresh (float): the threshold value to round value to 0.0 and 1.0, default is 0.5.

        """
        super().__init__(keys)
        if not isinstance(output_postfixes, (list, tuple)):
            raise ValueError("must specify key postfixes to store converted data.")
        if len(keys) != len(output_postfixes):
            raise ValueError("expected output items should match input data.")
        self.output_postfixes = output_postfixes
        self.converter = ConvertForMetrics(add_sigmoid, add_softmax, add_argmax, to_onehot_y_pred,
                                           to_onehot_y, n_classes, round_values, logit_thresh)

    def __call__(self, data):
        d = dict(data)
        y_pred_key, y_key = self.keys[0], self.keys[1] if len(self.keys) > 1 else None
        y_pred, y = d[y_pred_key], d[y_key] if y_key is not None else None
        y_pred, y = self.converter(y_pred, y)
        d[f"{y_pred_key}_{self.output_postfixes[0]}"] = y_pred
        if y_key is not None:
            d[f"{y_key}_{self.output_postfixes[1]}"] = y
        return d


ConvertForMetricsD = ConvertForMetricsDict = ConvertForMetricsd
SplitChannelD = SplitChannelDict = SplitChanneld
