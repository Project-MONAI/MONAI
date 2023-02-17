import unittest

import numpy as np

import torch

from monai.transforms.transform import LazyTrait, MultiSampleTrait, RandomizableTrait
from monai.transforms import ResizeWithPadOrCrop, CastToType
from monai.transforms import LoadImage, EnsureChannelFirst

from monai.transforms.spatial.array import (
    Flip,
    Rotate,
    Rotate90,
    Spacing,
    Zoom,
)
from monai.transforms.lazy.array import (
    ApplyTransforms,
    CachedTransform,
    CacheMechanism,
    MultiSampleTransform
)
from monai.transforms.utility.compose_compiler import ComposeCompiler


class MemoryCacheMechanism(CacheMechanism):
    """
    Enable testing of the CachedTransform with a simple memory-based key-value store
    """
    def __init__(
            self,
            max_count: int
    ):
        self.max_count = max_count
        self.contents = dict()
        self.order = list()

    def try_fetch(
            self,
            key
    ):
        if key in self.contents:
            return True, self.contents[key]

        return False, None

    def store(
            self,
            key,
            value
    ):
        if key in self.contents:
            self.contents[key] = value
        else:
            if len(self.contents) >= self.max_count:
                last = self.order.pop()
                del self.contents[last]

            self.contents[key] = value
            self.order.append(key)


class TestUtilityTransforms(unittest.TestCase):

    def test_cached_transform(self):

        def generate_noise(shape):
            def _inner(*args, **kwargs):
                return np.random.normal(size=shape)
            return _inner

        ct = CachedTransform(transforms=generate_noise((1, 16, 16)),
                             cache=MemoryCacheMechanism(4))

        first = ct("foo")
        second = ct("foo")
        third = ct("bar")

        self.assertIs(first, second)
        self.assertIsNot(first, third)

    def test_compile_caching(self):
        class NotRandomizable:
            def __init__(self, name):
                self.name = name

            def __repr__(self):
                return f"NR<{self.name}>"

        class Randomizable(RandomizableTrait):
            def __init__(self, name):
                self.name = name

            def __repr__(self):
                return f"R<{self.name}>"

        a = NotRandomizable("a")
        b = NotRandomizable("b")
        c = Randomizable("c")
        d = Randomizable("d")
        e = NotRandomizable("e")

        source_transforms = [a, b, c, d, e]

        cc = ComposeCompiler()

        actual = cc.compile_caching(source_transforms, CacheMechanism())

        self.assertIsInstance(actual[0], CachedTransform)
        self.assertEqual(len(actual[0].transforms), 2)
        self.assertTrue(actual[0].transforms[0], a)
        self.assertTrue(actual[0].transforms[1], b)
        self.assertTrue(actual[1], c)
        self.assertTrue(actual[2], d)
        self.assertTrue(actual[3], e)

    def test_compile_multisampling(self):

        class NotMultiSampling:
            def __init__(self, name):
               self.name = name

            def __repr__(self):
                return f"NMS<{self.name}>"

        class MultiSampling(MultiSampleTrait):
            def __init__(self, name):
                self.name = name

            def __repr__(self):
                return f"MS<{self.name}>"

        a = NotMultiSampling("a")
        b = NotMultiSampling("b")
        c = MultiSampling("c")
        d = NotMultiSampling("d")
        e = MultiSampling("e")
        f = NotMultiSampling("f")

        source_transforms = [a, b, c, d, e, f]

        cc = ComposeCompiler()

        actual = cc.compile_multisampling(source_transforms)

        self.assertEqual(actual[0], a)
        self.assertEqual(actual[1], b)
        self.assertIsInstance(actual[2], MultiSampleTransform)
        self.assertEqual(actual[2].multi_sample, c)
        self.assertEqual(len(actual[2].transforms), 2)
        self.assertEqual(actual[2].transforms[0], d)
        self.assertIsInstance(actual[2].transforms[1], MultiSampleTransform)
        self.assertEqual(actual[2].transforms[1].multi_sample, e)
        self.assertEqual(len(actual[2].transforms[1].transforms), 1)
        self.assertEqual(actual[2].transforms[1].transforms[0], f)

    def test_compile_lazy_resampling(self):
        class NotLazy:
            def __init__(self, name):
               self.name = name

            def __repr__(self):
                return f"NL<{self.name}>"

        class Lazy(LazyTrait):
            def __init__(self, name):
                self.name = name

            def __repr__(self):
                return f"L<{self.name}>"

        a = NotLazy("a")
        b = Lazy("b")
        c = Lazy("c")
        d = NotLazy("d")
        e = Lazy("e")
        f = Lazy("f")

        source_transforms = [a, b, c, d, e, f]

        cc = ComposeCompiler()

        actual = cc.compile_lazy_resampling(source_transforms)

        self.assertEqual(actual[0], a)
        self.assertEqual(actual[1], b)
        self.assertEqual(actual[2], c)
        self.assertIsInstance(actual[3], ApplyTransforms)
        self.assertEqual(actual[4], d)
        self.assertEqual(actual[5], e)
        self.assertEqual(actual[6], f)
        self.assertIsInstance(actual[7], ApplyTransforms)


class TestRealPipelineScenarios(unittest.TestCase):

    TEST_CASES = [
        (
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                # Orientation("RPS"),
                Spacing(pixdim=(1.2, 1.01, 0.9), mode="bilinear", dtype=np.float32),
                Flip(spatial_axis=[1, 2]),
                Rotate90(spatial_axes=(1, 2)),
                Zoom(zoom=0.75, keep_size=True),
                Rotate(angle=(np.pi, 0, 0), mode="bilinear", align_corners=True, dtype=np.float64),
                # RandAffine(prob=0.5, rotate_range=np.pi, mode="nearest"),
                ResizeWithPadOrCrop(100),
                CastToType(dtype=torch.uint8),
            ], (7,)
        ),
        (
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                # Orientation("RPS"),
                Spacing(pixdim=(1.2, 1.01, 0.9), mode="bilinear", dtype=np.float32, lazy_evaluation=False),
                Flip(spatial_axis=[1, 2]),
                Rotate90(spatial_axes=(1, 2), lazy_evaluation=False),
                Zoom(zoom=0.75, keep_size=True),
                Rotate(angle=(np.pi, 0, 0), mode="bilinear", align_corners=True, dtype=np.float64),
                # RandAffine(prob=0.5, rotate_range=np.pi, mode="nearest"),
                ResizeWithPadOrCrop(100),
                CastToType(dtype=torch.uint8),
            ], (7, )
        ),
    ]

    def test_compose_compiler_cases(self):
        for i_c, c in enumerate(self.TEST_CASES):
            self._test_compose_compiler(*c)

    def _test_compose_compiler(self, transforms, apply_locations):
        comp = ComposeCompiler()
        actual = comp.compile_lazy_resampling(transforms)
        offset = 0
        for a in actual:
            print(a)

        t = 0
        for a in range(len(actual)):
            if a not in apply_locations:
                self.assertIs(transforms[t], actual[a])
                t += 1
            else:
                self.assertIsInstance(actual[a], ApplyTransforms)
