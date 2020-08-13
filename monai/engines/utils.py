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

from typing import Dict, List, Optional, Sequence, Tuple

import torch


class CommonKeys:
    """
    A set of common keys for dictionary based supervised training process.
    `IMAGE` is the input image data.
    `LABEL` is the training or evaluation label of segmentation or classification task.
    `PRED` is the prediction data of model output.
    `LOSS` is the loss value of current iteration.
    `INFO` is some useful information during training or evaluation, like loss value, etc.

    """

    IMAGE = "image"
    LABEL = "label"
    PRED = "pred"
    LOSS = "loss"


class GanKeys:
    """
    A set of common keys for generative adversarial networks.
    """

    REALS = "reals"
    FAKES = "fakes"
    LATENTS = "latents"
    GLOSS = "g_loss"
    DLOSS = "d_loss"


def get_devices_spec(devices: Optional[Sequence[torch.device]] = None) -> List[torch.device]:
    """
    Get a valid specification for one or more devices. If `devices` is None get devices for all CUDA devices available.
    If `devices` is and zero-length structure a single CPU compute device is returned. In any other cases `devices` is
    returned unchanged.

    Args:
        devices: list of devices to request, None for all GPU devices, [] for CPU.

    Raises:
        RuntimeError: When all GPUs are selected (``devices=None``) but no GPUs are available.

    Returns:
        list of torch.device: list of devices.

    """
    if devices is None:
        devices = [torch.device(f"cuda:{d:d}") for d in range(torch.cuda.device_count())]

        if len(devices) == 0:
            raise RuntimeError("No GPU devices available.")

    elif len(devices) == 0:
        devices = [torch.device("cpu")]

    else:
        devices = list(devices)

    return devices


def default_prepare_batch(batchdata: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    assert isinstance(batchdata, dict), "default prepare_batch expects dictionary input data."
    if CommonKeys.LABEL in batchdata:
        return (batchdata[CommonKeys.IMAGE], batchdata[CommonKeys.LABEL])
    else:
        return (batchdata[CommonKeys.IMAGE], None)


def default_gan_prepare_batch(batchdata: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Prepares batchdata for GAN Discriminator training. Sends Tensor to executing device.

    Args:
        batchdata: Dictionary data returned by DataLoader
        devices: torch.device to store tensor for execution

    Raises:
        AssertionError: If input data is not a dictionary.
        RuntimeError: If dictionary does not have CommonKeys.IMAGE or GanKeys.REALS key.

    Returns:
        Batch size tensor of images on device.
    """
    assert isinstance(batchdata, dict), "default prepare_batch expects dictionary input data."
    if GanKeys.REALS in batchdata:
        data = batchdata[GanKeys.REALS]
    elif CommonKeys.IMAGE in batchdata:
        data = batchdata[CommonKeys.IMAGE]
    else:
        raise RuntimeError("default gan_prepare_batch expects '%s' or '%s' key." % (CommonKeys.IMAGE, GanKeys.REALS))
    return data.to(device)


def default_make_latent(
    num_latents: int, latent_size: int, device: torch.device, batchdata: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Prepares a latent code for GAN Generator training. Sends Tensor to executing device.
    If Generator needs additional input from Dataloader, override this func and process batchdata.

    Args:
        num_latents: number of latent codes to generate (typically batchsize)
        latent_size: size of latent code for Generator input
        device: torch.device to store tensor for execution
        batchdata: Minibatch from dataloader, ignored by default.

    Returns:
        Randomly generated latent codes.
    """
    return torch.randn(num_latents, latent_size).to(device)
