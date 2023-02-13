import unittest

from monai.transforms.utility import randomizer as rnd
from monai.transforms.utility.randomizer import ContinuousRandomizer
from .utils.fake_random import FakeRandomState


class TestContinuousRandomizer(unittest.TestCase):

    TEST_CASES = [
        (1.0, 1.0, 1.0, 1.0, None,
         FakeRandomState((('uniform', float(0.5)), ('uniform', float(1.0)))), 1.0),
        (0.5, 1.5, 0.4, 1.0, None,
         FakeRandomState((('uniform', float(0.5)), ('uniform', float(1.45)))), 1.0),
        (0.5, 1.5, 0.51, 1.0, None,
         FakeRandomState((('uniform', float(0.5)), ('uniform', float(1.45)))), 1.45,),
    ]

    def test_continuous_randomizer_cases(self):
        for i_c, c in enumerate(self.TEST_CASES):
            with self.subTest(i_c):
                self._test_continuous_randomizer(*c)

    def _test_continuous_randomizer(self, min_val, max_val, prob, default, seed, state, expected):

        r = ContinuousRandomizer(min_val, max_val, prob, default, seed, state)
        actual = r.sample()
        self.assertEqual(actual, expected)
