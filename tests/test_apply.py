import unittest

from monai.transforms import apply


class TestApply(unittest.TestCase):

    def _test_apply_impl(self):
        result = apply(None)
