from __future__ import annotations

import logging

import torch
import pytest

from monai.losses import PerceptualLoss

import time

@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("input_shape", [(2,1, 128,128,128)])
def test_perceptual_loss_speed(use_compile, device, input_shape):

    # Create random input tensors
    input_tensor = torch.rand(input_shape).to(device)
    target_tensor = torch.rand(input_shape).to(device)

    # Initialize the PerceptualLoss with the specified use_compile option
    loss_fn = PerceptualLoss(spatial_dims=len(input_shape) - 2, network_type="vgg", use_compile=use_compile).to(device)
    if device == "cuda":
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete before timing
        r = 50
    else:
        r = 1
    start_time = time.time()
    for _ in range(r):
        iteration_start = time.time()
        loss_value = loss_fn(input_tensor, target_tensor)
        iteration_end = time.time()
        logging.warning(f"Iteration time {iteration_end - iteration_start:.6f} seconds")
    end_time = time.time()

    logging.warning(f"PerceptualLoss forward pass time (use_compile={use_compile}, device={device}, input_shape={input_shape}): {end_time - start_time:.6f} seconds")