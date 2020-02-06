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


import math
import numpy as np
import torch
from monai.data.transforms import ImageEndPadder


def sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, device):
    """Use SlidingWindow method to execute inference.

    Args:
        roi_size (list, tuple): the window size to execute SlidingWindow inference.
        sw_batch_size (int): the batch size to run window slices.
        predictor: a portion from numpy.lib.arraypad.pad is copied below.
        device: on which device to execute model inference, cpu or gpu.

    Note:
        must be channel first, support both 2D and 3D.
        input data must have batch dim.
        execute on 1 image/per inference, run a batch of window slices of 1 input image.
    """
    num_spatial_dims = len(inputs.shape) - 2
    assert len(roi_size) == num_spatial_dims, 'roi_size {} does not match input dims.'.format(roi_size)

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    # TODO: Enable batch sizes > 1 in future.
    assert batch_size == 1, "input batch size must be 1"

    # in case that image size is smaller than roi size
    is_oversized = False
    original_image_size = [image_size[i] for i in range(num_spatial_dims)]

    for i in range(num_spatial_dims):
        if int(roi_size[i]) > image_size[i]:
            is_oversized = True
            break

    if is_oversized:
        for i in range(num_spatial_dims):
            image_size[i] = max(image_size[i], roi_size[i])

        padder = ImageEndPadder(roi_size, 'constant')
        inputs = padder(inputs)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims)
    scan_num = [int(math.ceil(float(image_size[i]) / scan_interval[i])) for i in range(num_spatial_dims)]

    # Store all slices in list.
    slices = []
    if num_spatial_dims == 3:
        for i in range(scan_num[0]):
            start_i = i * scan_interval[0]
            start_i -= max(start_i + roi_size[0] - image_size[0], 0)
            slice_i = slice(start_i, start_i + roi_size[0])

            for j in range(scan_num[1]):
                start_j = j * scan_interval[1]
                start_j -= max(start_j + roi_size[1] - image_size[1], 0)
                slice_j = slice(start_j, start_j + roi_size[1])

                for k in range(0, scan_num[2]):
                    start_k = k * scan_interval[2]
                    start_k -= max(start_k + roi_size[2] - image_size[2], 0)
                    slice_k = slice(start_k, start_k + roi_size[2])
                    slices.append((slice_i, slice_j, slice_k))
    else:
        for i in range(scan_num[0]):
            start_i = i * scan_interval[0]
            start_i -= max(start_i + roi_size[0] - image_size[0], 0)
            slice_i = slice(start_i, start_i + roi_size[0])

            for j in range(scan_num[1]):
                start_j = j * scan_interval[1]
                start_j -= max(start_j + roi_size[1] - image_size[1], 0)
                slice_j = slice(start_j, start_j + roi_size[1])
                slices.append((slice_i, slice_j))

    buffered_requests = []
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        input_slices = []
        for curr_index in slice_index_range:
            if num_spatial_dims == 3:
                slice_i, slice_j, slice_k = slices[curr_index]
                input_slices.append(inputs[0, :, slice_i, slice_j, slice_k])
            else:
                slice_i, slice_j = slices[curr_index]
                input_slices.append(inputs[0, :, slice_i, slice_j])
        buffered_requests.append(np.stack(input_slices))

    # Perform predictions
    output_rois = list()
    for data in buffered_requests:
        output_rois.append(predictor(data))

    output_classes = output_rois[0].shape[1]
    output_shape = [batch_size, output_classes] + list(image_size)

    # allocate memory to store the full output and the count for overlapping parts
    output_dict = torch.zeros(output_shape, dtype=torch.float32, device=device)
    count_dict = torch.zeros(output_shape, dtype=torch.float32, device=device)

    window_index = 0
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        output_roi = output_rois[window_index]
        window_index += 1

        # store the result in the proper location of the full output
        for curr_index in slice_index_range:
            if num_spatial_dims == 3:
                slice_i, slice_j, slice_k = slices[curr_index]
                output_dict[0, :, slice_i, slice_j, slice_k] += \
                    output_roi[curr_index - slice_index, :]
                count_dict[0, :, slice_i, slice_j, slice_k] += 1
            else:
                slice_i, slice_j = slices[curr_index]
                output_dict[0, :, slice_i, slice_j] += \
                    output_roi[curr_index - slice_index, :]
                count_dict[0, :, slice_i, slice_j] += 1

    # account for any overlapping sections
    output_dict /= count_dict

    # in case that image size is smaller than roi size
    if is_oversized:
        new_output_dict = list()
        if num_spatial_dims == 3:
            new_output_dict = output_dict[:, :, :original_image_size[0],
                                          :original_image_size[1], :original_image_size[2]]
        else:
            new_output_dict = output_dict[:, :, :original_image_size[0], :original_image_size[1]]

        output_dict = new_output_dict

    return output_dict


def _get_scan_interval(image_size, roi_size, num_spatial_dims):
    assert (len(image_size) == num_spatial_dims), 'image coord different from spatial dims.'
    assert (len(roi_size) == num_spatial_dims), 'roi coord different from spatial dims.'

    scan_interval = [1 for _ in range(num_spatial_dims)]
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval[i] = int(roi_size[i])
        else:
            # this means that it's r-16 (if r>=64) and r*0.75 (if r<=64)
            scan_interval[i] = int(max(roi_size[i] - 16, roi_size[i] * 0.75))
    return scan_interval
