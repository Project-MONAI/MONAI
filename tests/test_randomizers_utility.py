import unittest

from monai.transforms.utility import randomizer as rnd
from monai.transforms.utility.randomizer import ContinuousRandomizer
from .utilities.fake_random import FakeRandomState

FRS = FakeRandomState


def u_c(v):
    return ('uniform', float(v))


class TestContinuousRandomizer(unittest.TestCase):

    TEST_CASES = [
        (1.0, 1.0, 1.0, 1.0, None, FRS((u_c(0.5), u_c(1.0))), 1.0),
        (0.5, 1.5, 0.4, 1.0, None, FRS((u_c(0.5), u_c(1.45))), 1.0),
        (0.5, 1.5, 0.5, 1.0, None, FRS((u_c(0.5), u_c(1.45))), 1.45),
        (0.5, 1.5, 0.6, 1.0, None, FRS((u_c(0.5), u_c(1.45))), 1.45),
        ((1.0, 1.0), (1.0, 1.0), 1.0, 1.0, None, FRS((u_c(0.5), u_c(1.0), u_c(1.0))), (1.0, 1.0)),
        ((0.5, 2.0), (0.5, 2.0), 0.4, 1.25, None, FRS((u_c(0.5), u_c(0.8), u_c(1.2))), (1.25, 1.25)),
        ((0.5, 2.0), (0.5, 2.0), 0.5, 1.25, None, FRS((u_c(0.5), u_c(0.8), u_c(1.2))), (0.8, 1.2)),
        ((0.5, 2.0), (0.5, 2.0), 0.6, 1.25, None, FRS((u_c(0.5), u_c(0.8), u_c(1.2))), (0.8, 1.2)),
    ]

    def test_continuous_randomizer_cases(self):
        for i_c, c in enumerate(self.TEST_CASES):
            with self.subTest(i_c):
                self._test_continuous_randomizer(*c)

    def _test_continuous_randomizer(self, min_val, max_val, prob, default, seed, state, expected):

        r = ContinuousRandomizer(min_val, max_val, prob, default, seed, state)
        actual = r.sample()
        self.assertEqual(actual, expected)
