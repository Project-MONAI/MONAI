import unittest

import numpy as np

from monai.transforms.croppad.functional import croppad


class TestCropPad(unittest.TestCase):

    TEST_CASES = [
        ((16, 16), (slice(9, 13), slice(7, 11))),
        ((16, 16), (slice(8, 13), slice(6, 11))),
        ((17, 17), (slice(9, 13), slice(7, 11))),
        ((17, 17), (slice(8, 13), slice(6, 11))),
    ]

    def test_croppad_functional_cases(self):
        for i_c, c in enumerate(self.TEST_CASES):
            with self.subTest(i_c):
                self._test_croppad_functional(*c)

    def _test_croppad_functional(self, img_shape, crop_spans):
        img = np.zeros((1,) + img_shape)
        actual = croppad(img, crop_spans, lazy_evaluation=True)
        # print(actual.pending_operations[0])
        print(img_shape, crop_spans, actual.pending_operations[0].matrix.data[:2, -1])
