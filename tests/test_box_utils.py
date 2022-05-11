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

import random
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.data.box_utils import (
    box_affine,
    box_area,
    box_center,
    box_center_dist,
    box_clip_to_image,
    box_convert_mode,
    box_convert_standard_mode,
    box_giou,
    box_interp,
    box_iou,
    box_pair_giou,
    center_in_boxes,
    convert_to_list,
    non_max_suppression,
    resize_boxes,
)
from monai.utils.type_conversion import convert_data_type
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    bbox = [[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]]
    image_size = [4, 4, 4]
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xyzwhd", "half": False},
            "xyzwhd",
            p([[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]]),
            p([0, 12, 12]),
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xyzwhd", "half": True},
            "xyzxyz",
            p([[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 3, 3], [0, 1, 1, 2, 3, 4]]),
            p([0, 12, 12]),
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xyzwhd", "half": False},
            "xxyyzz",
            p([[0, 0, 0, 0, 0, 0], [0, 2, 1, 3, 0, 3], [0, 2, 1, 3, 1, 4]]),
            p([0, 12, 12]),
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xyzxyz", "half": False},
            "xyzwhd",
            p([[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 1, 3], [0, 1, 1, 2, 1, 2]]),
            p([0, 6, 4]),
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xyzxyz", "half": True},
            "xyzxyz",
            p([[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]]),
            p([0, 6, 4]),
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xyzxyz", "half": False},
            "xxyyzz",
            p([[0, 0, 0, 0, 0, 0], [0, 2, 1, 2, 0, 3], [0, 2, 1, 2, 1, 3]]),
            p([0, 6, 4]),
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xxyyzz", "half": False},
            "xxyyzz",
            p([[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]]),
            p([0, 2, 1]),
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xxyyzz", "half": True},
            "xyzxyz",
            p([[0, 0, 0, 0, 0, 0], [0, 0, 2, 1, 2, 3], [0, 1, 2, 1, 2, 3]]),
            p([0, 2, 1]),
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xxyyzz", "half": False},
            "xyzwhd",
            p([[0, 0, 0, 0, 0, 0], [0, 0, 2, 1, 2, 1], [0, 1, 2, 1, 1, 1]]),
            p([0, 2, 1]),
        ]
    )


class TestCreateBoxList(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, input_data, mode2, expected_box, expected_area):
        bbox1 = convert_data_type(input_data["bbox"], dtype=np.float32)[0]
        mode1 = input_data["mode"]
        image_size = input_data["image_size"]
        half_bool = input_data["half"]

        # test float16
        if half_bool:
            bbox1 = convert_data_type(bbox1, dtype=np.float16)[0]
            expected_box = convert_data_type(expected_box, dtype=np.float16)[0]

        # test box_convert_mode, box_convert_standard_mode
        result2 = box_convert_mode(bbox1=bbox1, mode1=mode1, mode2=mode2)
        assert_allclose(result2, expected_box, type_test=True, device_test=True, atol=0.0)

        result1 = box_convert_mode(bbox1=result2, mode1=mode2, mode2=mode1)
        assert_allclose(result1, bbox1, type_test=True, device_test=True, atol=0.0)

        result_standard = box_convert_standard_mode(bbox=bbox1, mode=mode1)
        expected_box_standard = box_convert_standard_mode(bbox=expected_box, mode=mode2)
        assert_allclose(result_standard, expected_box_standard, type_test=True, device_test=True, atol=0.0)

        # test box_area, box_clip_to_image, convert_to_list
        assert_allclose(box_area(result_standard), expected_area, type_test=True, device_test=True, atol=0.0)

        result_standard_clip, _ = box_clip_to_image(result_standard, image_size, remove_empty=True)
        np.testing.assert_equal(
            isinstance(result_standard_clip, np.ndarray), isinstance(bbox1, np.ndarray), "numpy type"
        )
        np.testing.assert_equal(
            isinstance(result_standard_clip, torch.Tensor), isinstance(bbox1, torch.Tensor), "torch type"
        )
        result_area_clip = convert_to_list(box_area(result_standard_clip))
        expected_area_clip = list(filter(lambda num: num > 0, convert_to_list(expected_area)))
        assert_allclose(result_area_clip, expected_area_clip, type_test=True, device_test=True, atol=0.0)

        # test box_interp and box_affine, resize_boxes
        zoom = [random.uniform(0.5, 5), random.uniform(0.5, 2), random.uniform(0.5, 5)]
        new_size = [int(image_size[axis] * zoom[axis] + 0.5) for axis in range(3)]
        zoom = [new_size[axis] / float(image_size[axis]) for axis in range(3)]

        result_standard_interp = box_interp(bbox=result_standard, zoom=zoom)
        result_standard_resize = resize_boxes(bbox=result_standard, original_size=image_size, new_size=new_size)
        assert_allclose(result_standard_interp, result_standard_resize, type_test=True, device_test=True, atol=0.0)

        result_area_interp = box_area(result_standard_interp)
        expected_area_interp = expected_area * zoom[0] * zoom[1] * zoom[2]
        assert_allclose(result_area_interp, expected_area_interp, type_test=True, device_test=True, atol=0.5)

        affine = torch.diag(torch.Tensor(zoom + [1.0]))
        result_affine = box_affine(bbox=bbox1, affine=affine, mode=mode1)
        result_affine_standard = box_convert_standard_mode(bbox=result_affine, mode=mode1)
        assert_allclose(
            box_area(result_affine_standard), expected_area_interp, type_test=True, device_test=True, atol=0.5
        )

        # test box_center, center_in_boxes, box_center_dist
        result_standard_center = box_center(result_standard)
        expected_center = box_convert_mode(bbox1=bbox1, mode1=mode1, mode2="cccwhd")[:, :3]
        assert_allclose(result_standard_center, expected_center, type_test=True, device_test=True, atol=0.0)

        center = expected_center
        center[2, :] += 10
        result_center_in_boxes = center_in_boxes(center=center, bbox=result_standard)
        assert_allclose(result_center_in_boxes, np.array([False, True, False]), type_test=False)

        center_dist, _, _ = box_center_dist(bbox1=result_standard[1:2, :], bbox2=result_standard[1:1, :])
        assert_allclose(center_dist, np.array([[]]), type_test=False)
        center_dist, _, _ = box_center_dist(bbox1=result_standard[1:2, :], bbox2=result_standard[1:2, :])
        assert_allclose(center_dist, np.array([[0.0]]), type_test=False)
        center_dist, _, _ = box_center_dist(bbox1=result_standard[0:1, :], bbox2=result_standard[0:1, :])
        assert_allclose(center_dist, np.array([[0.0]]), type_test=False)

        # test box_iou
        IOU_METRICS: Tuple[Callable] = (box_iou, box_giou)  # type: ignore
        for p in IOU_METRICS:
            self_iou = p(bbox1=result_standard[1:2, :], bbox2=result_standard[1:1, :])
            assert_allclose(self_iou, np.array([[]]), type_test=False)

            self_iou = p(bbox1=result_standard[1:2, :], bbox2=result_standard[1:2, :])
            assert_allclose(self_iou, np.array([[1.0]]), type_test=False)

        self_iou = box_pair_giou(bbox1=result_standard[1:1, :], bbox2=result_standard[1:1, :])
        assert_allclose(self_iou, np.array([]), type_test=False)

        self_iou = box_pair_giou(bbox1=result_standard[1:2, :], bbox2=result_standard[1:2, :])
        assert_allclose(self_iou, np.array([1.0]), type_test=False)

        # test non_max_suppression
        nms_box = non_max_suppression(
            bbox=result_standard, scores=bbox1[:, 1] / 2.0, nms_thresh=1.0, box_overlap_metric="iou"
        )
        assert_allclose(nms_box, [1, 2, 0], type_test=False)

        nms_box = non_max_suppression(
            bbox=result_standard, scores=bbox1[:, 1] / 2.0, nms_thresh=-0.1, box_overlap_metric="iou"
        )
        assert_allclose(nms_box, [1], type_test=False)


if __name__ == "__main__":
    unittest.main()
