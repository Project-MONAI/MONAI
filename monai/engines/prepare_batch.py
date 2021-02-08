from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from ..utils import ensure_tuple
from .utils import default_prepare_batch


class PrepareBatch(ABC):
    """
    Base class of customized prepare_batch in the trainer or evaluator workflows.
    It takes the data of current batch, target device and non_blocking flag as input.
    Default to run the default_prepare_batch of MONAI, which is consistent with ignite:
    https://github.com/pytorch/ignite/blob/v0.4.2/ignite/engine/__init__.py#L28.

    Note: if it is designed to be the last component in the prepare_batch chain,
        it should return either (image, label(optional)) or (image, label(optional), tuple, dict).
        the `tuple` and `dict` here are the `*args **kwargs` parameters for the network.
        otherwise, it should update the batchdata dict and return it directly.

    """

    @abstractmethod
    def __call__(
        self,
        batchdata: Dict[str, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
    ):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class PrepareBatchDefault(PrepareBatch):
    """
    Default prepare_batch method to return `image` and `label` only,
    it's consistent with MONAI `default_prerpare_batch` API.
    it should be the last component in the prepare_batch chain.

    """

    def __call__(
        self,
        batchdata: Dict[str, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
    ):
        return default_prepare_batch(batchdata, device, non_blocking)


class PrepareBatchExtraInput(PrepareBatch):
    """
    Customized prepare_batch for trainer or evalutor that support extra input data.
    it should be the last component in the prepare_batch chain.

    Args:
        extra_keys: if a string or list provided, every item is the key of extra data in current batch,
            and will pass the extra data to the network(*args) in order. if a dict provided,
            every {k, v} pair is the key of extra data in current batch, k the param name in network,
            v is the key of extra data in current batch, and will pass the {k1: d1, k2: d2, ...}
            dict to the network(**kwargs).

    """

    def __init__(self, extra_keys: Union[str, Sequence[str], Dict[str, str]]) -> None:
        self.extra_keys = extra_keys

    def __call__(
        self,
        batchdata: Dict[str, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
    ):
        image, label = default_prepare_batch(batchdata, device, non_blocking)
        args = list()
        kwargs = dict()

        def _get_data(key: str):
            data = batchdata[key]
            return data.to(device=device, non_blocking=non_blocking) if torch.is_tensor(data) else data

        if isinstance(self.extra_keys, (str, list, tuple)):
            for k in ensure_tuple(self.extra_keys):
                args.append(_get_data(k))
        elif isinstance(self.extra_keys, dict):
            for k, v in self.extra_keys.items():
                kwargs.update({k: _get_data(v)})

        return image, label, tuple(args), kwargs


class PrepareBatchShuffle(PrepareBatch):
    """
    Prepare shuffled batches by random permutation of samples within each batch.

    """

    def __call__(
        self,
        batchdata: Dict[str, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:

        images, labels = default_prepare_batch(batchdata, device, non_blocking)
        idx_rand = torch.randperm(len(labels)).to(device=device, non_blocking=non_blocking)

        return images[idx_rand], labels[idx_rand]
