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


def get_devices_spec(devices=None):
    """
    Get a valid specification for one or more devices. If `devices` is None get devices for all CUDA devices available.
    If `devices` is and zero-length structure a single CPU compute device is returned. In any other cases `devices` is
    returned unchanged.

    Args:
        devices (list, optional): list of devices to request, None for all GPU devices, [] for CPU.

    Returns:
        list of torch.device: list of devices.

    Raises:
        ValueError: No GPU devices available

    """
    if devices is None:
        devices = [torch.device(f"cuda:{d:d}") for d in range(torch.cuda.device_count())]

        if len(devices) == 0:
            raise ValueError("No GPU devices available")

    elif len(devices) == 0:
        devices = [torch.device("cpu")]

    return devices


def default_prepare_batch(batchdata):
    assert isinstance(batchdata, dict), "default prepare_batch expects dictionary input data."
    return (
        (batchdata[CommonKeys.IMAGE], batchdata[CommonKeys.LABEL])
        if CommonKeys.LABEL in batchdata
        else (batchdata[CommonKeys.IMAGE], None)
    )
