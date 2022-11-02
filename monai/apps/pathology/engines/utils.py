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

from typing import Dict, Optional, Sequence, Union

import torch

from monai.engines import PrepareBatch, PrepareBatchExtraInput
from monai.utils import ensure_tuple
from monai.utils.enums import HoVerNetBranch

__all__ = ["PrepareBatchWithDictLabel"]


class PrepareBatchWithDictLabel(PrepareBatch):
    """
    Customized prepare batch callable for trainers or evaluators which support label to be a dictionary.
    Extra items are specified by the `extra_keys` parameter and are extracted from the input dictionary (ie. the batch).
    This assumes label is a dictionary.

    Args:
        extra_keys: If a sequence of strings is provided, values from the input dictionary are extracted from
            those keys and passed to the nework as extra positional arguments.
    """

    def __init__(self, extra_keys: Sequence[str]) -> None:
        if len(ensure_tuple(extra_keys)) != 2:
            raise ValueError(f"length of `extra_keys` should be 2, get {len(ensure_tuple(extra_keys))}")
        self.prepare_batch = PrepareBatchExtraInput(extra_keys)

    def __call__(
        self,
        batchdata: Dict[str, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        **kwargs,
    ):
        """
        Args `batchdata`, `device`, `non_blocking` refer to the ignite API:
        https://pytorch.org/ignite/v0.4.8/generated/ignite.engine.create_supervised_trainer.html.
        `kwargs` supports other args for `Tensor.to()` API.
        """
        image, _label, extra_label, _ = self.prepare_batch(batchdata, device, non_blocking, **kwargs)
        label = {HoVerNetBranch.NP: _label, HoVerNetBranch.NC: extra_label[0], HoVerNetBranch.HV: extra_label[1]}

        return image, label
