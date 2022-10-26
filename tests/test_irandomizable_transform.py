import unittest

import numpy as np

from monai.transforms.transform import IRandomizableTransform, RandomizableTransform


class InheritsInterface(IRandomizableTransform):
    pass


class InheritsImplementation(RandomizableTransform):

    def __call__(self, data):
        return data


class TestIRandomizableTransform(unittest.TestCase):

    def test_is_irandomizable(self):
        inst = InheritsInterface()
        self.assertIsInstance(inst, IRandomizableTransform)

    def test_set_random_state_default_impl(self):
        inst = InheritsInterface()
        with self.assertRaises(TypeError):
            inst.set_random_state(seed=0)

    def test_set_random_state_randomizable_transform(self):
        inst = InheritsImplementation()
        inst.set_random_state(0)
