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

from typing import List

import torch
import torch.distributed as dist

from monai.config import IgniteInfo
from monai.utils.module import min_version, optional_import

idist, has_ignite = optional_import("ignite", IgniteInfo.OPT_IMPORT_VERSION, min_version, "distributed")

__all__ = ["get_dist_device", "evenly_divisible_all_gather", "string_list_all_gather"]


def get_dist_device():
    """
    Get the expected target device in the native PyTorch distributed data parallel.
    For NCCL backend, return GPU device of current process.
    For GLOO backend, return CPU.
    For any other backends, return None as the default, tensor.to(None) will not change the device.

    """
    if dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        if backend == "gloo":
            return torch.device("cpu")
    return None


def evenly_divisible_all_gather(data: torch.Tensor, concat: bool = True):
    """
    Utility function for distributed data parallel to pad at first dim to make it evenly divisible and all_gather.
    The input data of every rank should have the same number of dimensions, only the first dim can be different.

    Note: If has ignite installed, will execute based on ignite distributed APIs, otherwise, if the native
    PyTorch distributed group initialized, will execute based on native PyTorch distributed APIs.

    Args:
        data: source tensor to pad and execute all_gather in distributed data parallel.
        concat: whether to concat the gathered list to be a Tensor, if False, return a list
            of Tensors, similar behavior as torch.distributed.all_gather(). default to True.

    Note:
        The input data on different ranks must have exactly same `dtype`.

    """
    if not isinstance(data, torch.Tensor):
        raise ValueError("input data must be PyTorch Tensor.")
    # data of all the ranks must have same number of dimensions
    ndims = data.ndimension()
    length: int = data.shape[0] if ndims > 0 else 1

    def _torch_all_gather(data: torch.Tensor) -> List[torch.Tensor]:
        """
        Implementation based on native PyTorch distributed data parallel APIs.

        """
        device = get_dist_device()
        orig_device = data.device
        data = data.to(device)
        data = data.unsqueeze(0) if ndims == 0 else data

        # make sure the data is evenly-divisible on multi-GPUs
        length_tensor = torch.as_tensor([length], device=device)
        all_lens = [torch.zeros_like(length_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(all_lens, length_tensor)
        all_lens_: List[int] = [int(i.item()) for i in all_lens]

        max_len: int = max(all_lens_)
        if length < max_len:
            size = [max_len - length] + list(data.shape[1:])
            data = torch.cat([data, data.new_full(size, 0)], dim=0)
        # all gather across all processes
        output = [torch.zeros_like(data) for _ in range(dist.get_world_size())]
        dist.all_gather(output, data)
        # remove the padding items, if all the input data doesn't have batch dim, squeeze the first dim
        return [(o.squeeze(0) if ndims == 0 else o[:l, ...]).to(orig_device) for o, l in zip(output, all_lens_)]

    def _ignite_all_gather(data: torch.Tensor) -> List[torch.Tensor]:
        """
        Implementation based on PyTorch ignite package, it can support more kinds of backends.

        """
        data = data.unsqueeze(0) if ndims == 0 else data
        # make sure the data is evenly-divisible on multi-GPUs
        all_lens: List[int] = idist.all_gather(length)
        max_len: int = max(all_lens)
        if length < max_len:
            size = [max_len - length] + list(data.shape[1:])
            data = torch.cat([data, data.new_full(size, 0)], dim=0)
        # all gather across all processes
        output = idist.all_gather(data)
        # delete the padding NaN items
        if ndims == 0:
            # if all the input data doesn't have batch dim, unbind to a list of 0-dim Tensors
            return list(torch.unbind(output, dim=0))
        return [output[i * max_len : i * max_len + l, ...] for i, l in enumerate(all_lens)]

    output: List[torch.Tensor]
    if has_ignite:
        if idist.get_world_size() <= 1:
            return data
        output = _ignite_all_gather(data=data)
    elif dist.is_available() and dist.is_initialized():
        if dist.get_world_size() <= 1:
            return data
        output = _torch_all_gather(data=data)
    else:
        return data

    return torch.cat(output, dim=0) if concat else output


def string_list_all_gather(strings: List[str], delimiter: str = "\t") -> List[str]:
    """
    Utility function for distributed data parallel to all gather a list of strings.
    Refer to the idea of ignite `all_gather(string)`:
    https://pytorch.org/ignite/v0.4.5/distributed.html#ignite.distributed.utils.all_gather.

    Note: If has ignite installed, will execute based on ignite distributed APIs, otherwise, if the native
    PyTorch distributed group initialized, will execute based on native PyTorch distributed APIs.

    Args:
        strings: a list of strings to all gather.
        delimiter: use the delimiter to join the string list to be a long string,
            then all gather across ranks and split to a list. default to "\t".

    """
    world_size: int = 1
    if has_ignite:
        world_size = idist.get_world_size()
    elif dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()

    if world_size <= 1:
        return strings

    joined = delimiter.join(strings)
    gathered = evenly_divisible_all_gather(torch.tensor(bytearray(joined, "utf-8"), dtype=torch.long), concat=False)
    gathered = [bytearray(g.tolist()).decode("utf-8").split(delimiter) for g in gathered]

    return [i for k in gathered for i in k]
