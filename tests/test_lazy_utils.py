import unittest

import numpy as np

from monai.transforms.utils import create_rotate, create_scale, create_flip, create_rotate_90, create_shear
from monai.transforms.lazy.utils import matrix_to_eulers, check_matrix, check_axes

import torch


class TestMatrixToEulers(unittest.TestCase):

    def test_matrix_to_eulers(self):

        print(np.linalg.norm(np.asarray([1.0, 1.0, 0.0])))
        print(np.linalg.norm(np.asarray([0.0, 0.0, 0.0])))

        arr = np.asarray(np.eye(4))
        result = matrix_to_eulers(arr)
        print(result)

        arr = np.asarray(create_rotate(3, (0, 0, torch.pi)))
        result = matrix_to_eulers(arr)
        print(result)

        arr = np.asarray(create_rotate(3, (0, 0, torch.pi / 2)))
        result = matrix_to_eulers(arr)
        print(result)


class TestCheckMatrix(unittest.TestCase):

    TEST_CASES = [
        (np.eye(4), (True, True)),
        (create_scale(3, (1.0, 1.0, 1.0)), (True, True)),
        (create_scale(3, (1.1, 1.0, 1.0)), (True, False)),
        (create_scale(3, (1.0, 1.1, 1.0)), (True, False)),
        (create_scale(3, (1.0, 1.0, 1.1)), (True, False)),
        (create_scale(3, (0.9, 1.0, 1.0)), (True, False)),
        (create_scale(3, (1.0, 0.9, 1.0)), (True, False)),
        (create_scale(3, (1.0, 1.0, 0.9)), (True, False)),
        (create_scale(3, (0.9, 0.9, 0.9)), (True, False)),
        (create_scale(3, (1.1, 1.1, 1.1)), (True, False)),
        (create_flip(3, 0), (True, True)),
        (create_flip(3, 1), (True, True)),
        (create_flip(3, 2), (True, True)),
        (create_flip(3, (0, 1)), (True, True)),
        (create_flip(3, (0, 2)), (True, True)),
        (create_flip(3, (1, 2)), (True, True)),
        (create_flip(3, (0, 1, 2)), (True, True)),
        (create_rotate_90(3, (1, 2), 0), (True, True)),
        (create_rotate_90(3, (1, 2), 1), (True, True)),
        (create_rotate_90(3, (1, 2), 2), (True, True)),
        (create_rotate_90(3, (1, 2), 3), (True, True)),
        (create_rotate(3, (0, 0, torch.pi / 2)), (True, True)),
        (create_rotate(3, (0, 0, torch.pi)), (True, True)),
        (create_rotate(3, (0, 0, 3 * torch.pi / 2)), (True, True)),
        (create_rotate(3, (0, 0, 2 * torch.pi)), (True, True)),
        (create_rotate(3, (0, 0, torch.pi / 4)), (False, True)),
        (create_shear(3, 2, 0.5), (False, False))
    ]

    def test_check_matrix_cases(self):
        for i_c, c in enumerate(self.TEST_CASES):
            with self.subTest(i_c):
                self._test_check_matrix(*c)

    def _test_check_matrix(self, matrix, expected):
        self.assertEqual(check_matrix(matrix), expected)

class TestCheckAxes(unittest.TestCase):

    TEST_CASES = [
        (np.eye(4), ((0, 1), (1, 1), (2, 1))),
        (create_rotate(3, (0, 0, torch.pi / 2)), ((1, -1), (0, 1), (2, 1))),
        (create_rotate(3, (0, torch.pi / 2, 0)), ((2, 1), (1, 1), (0, -1))),
        (create_rotate(3, (torch.pi / 2, 0, 0)), ((0, 1), (2, -1), (1, 1))),
        (create_rotate(3, (0, 0, torch.pi)), ((0, -1), (1, -1), (2, 1))),
        (create_rotate(3, (0, torch.pi, 0)), ((0, -1), (1, 1), (2, -1))),
        (create_rotate(3, (torch.pi, 0, 0)), ((0, 1), (1, -1), (2, -1))),
    ]
    def test_check_axes_cases(self):
        for i_c, c in enumerate(self.TEST_CASES):
            with self.subTest(f"{i_c}"):
                self._test_check_axes(*c)

    def _test_check_axes(self, matrix, expected):
        self.assertEqual(check_axes(matrix), expected)