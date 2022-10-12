import unittest

import math

import numpy as np


import torch

from monai.transforms.atmostonce import array as amoa
from monai.transforms.atmostonce.array import Rotate, CropPad
from monai.transforms.atmostonce.lazy_transform import compile_lazy_transforms
from monai.transforms.atmostonce.utils import value_to_tuple_range
from monai.utils import TransformBackends

from monai.transforms import Affined, Affine, Flip, RandSpatialCropSamplesd, RandRotated
from monai.transforms.atmostonce.functional import croppad, resize, rotate, zoom, spacing, flip
from monai.transforms.atmostonce.apply import Applyd, extents_from_shape, shape_from_extents, apply
from monai.transforms.atmostonce.dictionary import Rotated
from monai.transforms.compose import Compose
from monai.utils.enums import GridSampleMode, GridSamplePadMode
from monai.utils.mapping_stack import MatrixFactory

from monai.transforms.atmostonce.utility import CachedTransform, CacheMechanism


def get_img(size, dtype=torch.float32, offset=0):
    img = torch.zeros(size, dtype=dtype)
    if len(size) == 2:
        for j in range(size[0]):
            for i in range(size[1]):
                img[j, i] = i + j * size[0] + offset
    else:
        for k in range(size[0]):
            for j in range(size[1]):
                for i in range(size[2]):
                    img[k, j, i] = i + j * size[0] + k * size[0] * size[1]
    return np.expand_dims(img, 0)


def enumerate_results_of_op(results):
    if isinstance(results, dict):
        for k, v in results.items():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                print(k, v.shape, v[tuple(slice(0, 8) for _ in v.shape)])
            else:
                print(k, v)
    else:
        for ir, v in enumerate(results):
            if isinstance(v, (np.ndarray, torch.Tensor)):
                print(ir, v.shape, v[tuple(slice(0, 8) for _ in v.shape)])
            else:
                print(ir, v)


def matrices_nearly_equal(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("actual matrix does not match expected matrix size; "
                         f"{actual} vs {expected} respectively")


class TestLowLevel(unittest.TestCase):

    def test_extents_2(self):
        actual = extents_from_shape([1, 24, 32])
        expected = [np.asarray(v) for v in ((0, 0, 1), (0, 32, 1), (24, 0, 1), (24, 32, 1))]
        self.assertTrue(np.all([np.array_equal(a, e) for a, e in zip(actual, expected)]))

    def test_extents_3(self):
        actual = extents_from_shape([1, 12, 16, 8])
        expected = [np.asarray(v) for v in ((0, 0, 0, 1), (0, 0, 8, 1), (0, 16, 0, 1), (0, 16, 8, 1),
                                            (12, 0, 0, 1), (12, 0, 8, 1), (12, 16, 0, 1), (12, 16, 8, 1))]
        self.assertTrue(np.all([np.array_equal(a, e) for a, e in zip(actual, expected)]))

    def test_shape_from_extents(self):
        actual = shape_from_extents([np.asarray([-16, -20, 1]),
                                     np.asarray([-16, 20, 1]),
                                     np.asarray([16, -20, 1]),
                                     np.asarray([16, 20, 1])])
        print(actual)


    def test_compile_transforms(self):
        values = ["a", "b", ["c", ["d"], "e"], "f", ["g", "h"], "i"]
        result = compile_lazy_transforms(values)
        print(result)


class TestMappingStack(unittest.TestCase):

    def test_rotation_pi_by_2(self):

        fac = MatrixFactory(2, TransformBackends.NUMPY)
        mat = fac.rotate_euler(torch.pi / 2)
        expected = np.asarray([[0, -1, 0],
                               [1, 0, 0],
                               [0, 0, 1]])
        self.assertTrue(np.allclose(mat.matrix.matrix, expected))

    def test_rotation_pi_by_4(self):

        fac = MatrixFactory(2, TransformBackends.NUMPY)
        mat = fac.rotate_euler(torch.pi / 4)
        piby4 = math.cos(torch.pi / 4)
        expected = np.asarray([[piby4, -piby4, 0],
                               [piby4, piby4, 0],
                               [0, 0, 1]])
        self.assertTrue(np.allclose(mat.matrix.matrix, expected))

    def test_rotation_pi_by_8(self):
        fac = MatrixFactory(2, TransformBackends.NUMPY)
        mat = fac.rotate_euler(torch.pi / 8)
        cospi = math.cos(torch.pi / 8)
        sinpi = math.sin(torch.pi / 8)
        expected = np.asarray([[cospi, -sinpi, 0],
                               [sinpi, cospi, 0],
                               [0, 0, 1]])
        self.assertTrue(np.allclose(mat.matrix.matrix, expected))

    def scale_by_2(self):
        fac = MatrixFactory(2, TransformBackends.NUMPY)
        mat = fac.scale(2)
        expected = np.asarray([[2, 0, 0],
                               [0, 2, 0],
                               [0, 0, 1]])
        self.assertTrue(np.allclose(mat.matrix.matrix, expected))

    # TODO: turn into proper test
    def test_mult_matrices(self):

        fac = MatrixFactory(2, TransformBackends.NUMPY)
        matrix1 = fac.translate((-16, -16))
        matrix2 = fac.rotate_euler(torch.pi / 4)

        matrix12 = matrix1 @ matrix2
        matrix21 = matrix2 @ matrix1

        print("matrix12\n", matrix12.matrix.matrix)
        print("matrix21\n", matrix21.matrix.matrix)

        extents = extents_from_shape([1, 32, 32])

        print("matrix1")
        for e in extents:
            print("  ", e, matrix1.matrix.matrix @ e)
        print("matrix2")
        for e in extents:
            print("  ", e, matrix2.matrix.matrix @ e)
        print("matrix12")
        for e in extents:
            print("  ", e, matrix12.matrix.matrix @ e)
        print("matrix21")
        for e in extents:
            print("  ", e, matrix21.matrix.matrix @ e)


class TestFunctional(unittest.TestCase):

    def _test_functional_impl(self,
                              op,
                              image,
                              params,
                              expected_matrix):
        r_image, r_transform, r_metadata = op(image, **params)
        enumerate_results_of_op((r_image, r_transform, r_metadata))
        self.assertTrue(torch.allclose(r_transform, expected_matrix))

    # TODO: turn into proper test
    def test_spacing(self):
        kwargs = {
            "pixdim": (0.5, 0.6), "src_pixdim": (1.0, 1.0), "diagonal": False,
            "mode": "bilinear", "padding_mode": "border", "align_corners": None
        }
        expected_tx = torch.DoubleTensor([[2.0, 0.0, 0.0],
                                          [0.0, 1.66666667, 0.0],
                                          [0.0, 0.0, 1.0]])
        self._test_functional_impl(spacing, get_img((24, 32)), kwargs, expected_tx)


    # TODO: turn into proper test
    def test_resize(self):
        kwargs = {
            "spatial_size": (40, 40), "size_mode": "all",
            "mode": "bilinear", "align_corners": None
        }
        expected_tx = torch.DoubleTensor([[1.66666667, 0.0, 0.0],
                                          [0.0, 1.25, 0.0],
                                          [0.0, 0.0, 1.0]])
        self._test_functional_impl(resize, get_img((24, 32)), kwargs, expected_tx)


    # TODO: turn into proper test
    def test_rotate(self):
        kwargs = {
            "angle": torch.pi / 4, "keep_size": True,
            "mode": "bilinear", "padding_mode": "border"
        }
        expected_tx = torch.DoubleTensor([[0.70710678, -0.70710678, 0.0],
                                          [0.70710678, 0.70710678, 0.0],
                                          [0.0, 0.0, 1.0]])
        self._test_functional_impl(rotate, get_img((24, 32)), kwargs, expected_tx)


    def test_zoom(self):
        # results = zoom(np.zeros((1, 64, 64), dtype=np.float32),
        #                2,
        #                "bilinear",
        #                "zeros")
        # enumerate_results_of_op(results)
        kwargs = {
            "factor": 2, "mode": "nearest", "padding_mode": "border", "keep_size": True
        }
        expected_tx = torch.DoubleTensor([[0.5, 0.0, 0.0],
                                          [0.0, 0.5, 0.0],
                                          [0.0, 0.0, 1.0]])
        self._test_functional_impl(zoom, get_img((24, 32)), kwargs, expected_tx)


    def _check_matrix(self, actual, expected):
        np.allclose(actual, expected)

    def _test_rotate_90_impl(self, values, keep_dims, expected):
        results = rotate(np.zeros((1, 64, 64, 32), dtype=np.float32),
                         values,
                         keep_dims,
                         "bilinear",
                         "border")
        # enumerate_results_of_op(results)
        self._check_matrix(results[1], expected)

    def test_rotate_d0_r1(self):
        expected = np.asarray([[1, 0, 0, 0],
                               [0, 0, -1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1]])
        self._test_rotate_90_impl((torch.pi / 2, 0, 0), True, expected)

    def test_rotate_d0_r2(self):
        expected = np.asarray([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])
        self._test_rotate_90_impl((torch.pi, 0, 0), True, expected)

    def test_rotate_d0_r3(self):
        expected = np.asarray([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, -1, 0, 0],
                               [0, 0, 0, 1]])
        self._test_rotate_90_impl((3 * torch.pi / 2, 0, 0), True, expected)

    def test_rotate_d2_r1(self):
        expected = np.asarray([[0, -1, 0, 0],
                               [1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        self._test_rotate_90_impl((0, 0, torch.pi / 2), True, expected)

    def test_rotate_d2_r2(self):
        expected = np.asarray([[-1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        self._test_rotate_90_impl((0, 0, torch.pi), True, expected)

    def test_rotate_d2_r3(self):
        expected = np.asarray([[0, 1, 0, 0],
                               [-1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        self._test_rotate_90_impl((0, 0, 3 * torch.pi / 2), True, expected)

    def test_croppad_identity(self):
        img = get_img((16, 16)).astype(int)
        results = croppad(img,
                          (slice(0, 16), slice(0, 16)))
        enumerate_results_of_op(results)
        m = results[1].matrix.matrix
        print(m)
        result_size = results[2]['spatial_shape']
        a = Affine(affine=m,
                   padding_mode=GridSamplePadMode.ZEROS,
                   spatial_size=result_size)
        img_, _ = a(img)
        print(img_)

    def _croppad_impl(self, img_ext, slices, expected):
        img = get_img(img_ext).astype(int)
        results = croppad(img, slices)
        enumerate_results_of_op(results)
        m = results[1].matrix.matrix
        print(m)
        result_size = results[2]['spatial_shape']
        a = Affine(affine=m,
                   padding_mode=GridSamplePadMode.ZEROS,
                   spatial_size=result_size)
        img_, _ = a(img)
        if expected is None:
            print(img_.numpy())
        else:
            self.assertTrue(torch.allclose(img_, expected))

    def test_croppad_img_odd_crop_odd(self):
        expected = torch.as_tensor([[63., 64., 65., 66., 67., 68., 69.],
                                    [78., 79., 80., 81., 82., 83., 84.],
                                    [93., 94., 95., 96., 97., 98., 99.],
                                    [108., 109., 110., 111., 112., 113., 114.],
                                    [123., 124., 125., 126., 127., 128., 129.]])
        self._croppad_impl((15, 15), (slice(4, 9), slice(3, 10)), expected)

    def test_croppad_img_odd_crop_even(self):
        expected = torch.as_tensor([[63., 64., 65., 66., 67., 68.],
                                    [78., 79., 80., 81., 82., 83.],
                                    [93., 94., 95., 96., 97., 98.],
                                    [108., 109., 110., 111., 112., 113.]])
        self._croppad_impl((15, 15), (slice(4, 8), slice(3, 9)), expected)

    def test_croppad_img_even_crop_odd(self):
        expected = torch.as_tensor([[67., 68., 69., 70., 71., 72., 73.],
                                    [83., 84., 85., 86., 87., 88., 89.],
                                    [99., 100., 101., 102., 103., 104., 105.],
                                    [115., 116., 117., 118., 119., 120., 121.],
                                    [131., 132., 133., 134., 135., 136., 137.]])
        self._croppad_impl((16, 16), (slice(4, 9), slice(3, 10)), expected)

    def test_croppad_img_even_crop_even(self):
        expected = torch.as_tensor([[67., 68., 69., 70., 71., 72.],
                                    [83., 84., 85., 86., 87., 88.],
                                    [99., 100., 101., 102., 103., 104.],
                                    [115., 116., 117., 118., 119., 120.]])
        self._croppad_impl((16, 16), (slice(4, 8), slice(3, 9)), expected)

    def _test_flip_impl(self, dims, spatial_axis, expected, verbose=False):
        if dims == 2:
            img = get_img((32, 32))
        else:
            img = get_img((32, 32, 8))

        actual = flip(img, spatial_axis=spatial_axis)
        if verbose:
            print("expected\n", expected)
            print("actual\n", actual[1])
        self.assertTrue(np.allclose(expected, actual[1]))

    def test_flip(self):

        tests = [
            (2, None, {(0, 0): -1, (1, 1): -1}),
            (2, 0, {(0, 0): -1}),
            (2, 1, {(1, 1): -1}),
            (2, (0,), {(0, 0): -1}),
            (2, (1,), {(1, 1): -1}),
            (2, (0, 1), {(0, 0): -1, (1, 1): -1}),
            (3, None, {(0, 0): -1, (1, 1): -1, (2, 2): -1}),
            (3, 0, {(0, 0): -1}),
            (3, 1, {(1, 1): -1}),
            (3, 2, {(2, 2): -1}),
            (3, (0,), {(0, 0): -1}),
            (3, (1,), {(1, 1): -1}),
            (3, (2,), {(2, 2): -1}),
            (3, (0, 1), {(0, 0): -1, (1, 1): -1}),
            (3, (0, 2), {(0, 0): -1, (2, 2): -1}),
            (3, (1, 2), {(1, 1): -1, (2, 2): -1}),
            (3, (0, 1, 2), {(0, 0): -1, (1, 1): -1, (2, 2): -1}),
        ]

        for t in tests:
            with self.subTest(f"{t}"):
                expected = np.eye(t[0] + 1)
                for ke, kv in t[2].items():
                    expected[ke] = kv
                self._test_flip_impl(t[0], t[1], expected)


class TestArrayTransforms(unittest.TestCase):

    # TODO: amo: add tests for matrix and result size
    def test_croppad(self):
        img = get_img((15, 15)).astype(int)
        results = croppad(img, (slice(4, 8), slice(3, 9)))
        enumerate_results_of_op(results)
        m = results[1].matrix.matrix
        # print(m)
        result_size = results[2]['spatial_shape']
        a = Affine(affine=m,
                   padding_mode=GridSamplePadMode.ZEROS,
                   spatial_size=result_size)
        img_, _ = a(img)
        # print(img_.numpy())

    def test_apply(self):
        img = get_img((16, 16))
        r = Rotate(torch.pi / 4,
                   keep_size=False,
                   mode="bilinear",
                   padding_mode="zeros",
                   lazy_evaluation=True)
        c = CropPad((slice(4, 12), slice(6, 14)),
                    lazy_evaluation=True)

        img_r = r(img)
        cur_op = img_r.peek_pending_transform()
        img_rc = c(img_r,
                   shape_override=cur_op.metadata.get("shape_override", None))

        img_rca = apply(img_rc)

    def test_rand_rotate(self):
        r = amoa.RandRotate((-torch.pi / 4, torch.pi / 4),
                            prob=0.0,
                            keep_size=True,
                            mode="bilinear",
                            padding_mode="border",
                            align_corners=False)
        img = np.zeros((1, 32, 32), dtype=np.float32)
        results = r(img)
        enumerate_results_of_op(results)
        enumerate_results_of_op(results.pending_transforms[-1].metadata)

    def test_rotate_apply_not_lazy(self):
        r = amoa.Rotate(-torch.pi / 4,
                        mode="bilinear",
                        padding_mode="border",
                        keep_size=False)
        data = get_img((32, 32))
        data = r(data)
        # data = apply(data)
        print(data.shape)
        print(data)

    def test_rotate_apply_lazy(self):
        r = amoa.Rotate(-torch.pi / 4,
                        mode="bilinear",
                        padding_mode="border",
                        keep_size=False)
        r.lazy_evaluation = True
        data = get_img((32, 32))
        data = r(data)
        data = apply(data)
        expected = torch.DoubleTensor([[0.70710677, 0.70710677, 0.0, -15.61269784],
                                       [-0.70710677, 0.70710677, 0.0, 15.5],
                                       [0.0, 0.0, 1.0, 0.0],
                                       [0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(torch.allclose(expected, data.affine))

    def test_zoom_apply_lazy(self):
        r = amoa.Zoom(2,
                      mode="bilinear",
                      padding_mode="border",
                      keep_size=False)
        r.lazy_evaluation = True
        data = get_img((32, 32))
        data = r(data)
        data = apply(data)
        expected = torch.DoubleTensor([[0.5, 0.0, 0.0, 11.75],
                                      [0.0, 0.5, 0.0, 11.75],
                                      [0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(torch.allclose(expected, data.affine))

    def test_crop_then_rotate_apply_lazy(self):
        data = get_img((32, 32))
        print(data.shape)

        lc1 = amoa.CropPad(lazy_evaluation=True,
                           padding_mode="zeros")
        lr1 = amoa.Rotate(torch.pi / 4,
                          keep_size=False,
                          padding_mode="zeros",
                          lazy_evaluation=False)
        datas = []
        datas.append(data)
        data1 = lc1(data, slices=(slice(0, 16), slice(0, 16)))
        datas.append(data1)
        data2 = lr1(data1)
        datas.append(data2)


class TestDictionaryTransforms(unittest.TestCase):

    def test_rotate_numpy(self):
        r = Rotated(('image', 'label'), [0.0, 1.0, 0.0])

        d = {
            'image': np.zeros((1, 64, 64, 32), dtype=np.float32),
            'label': np.ones((1, 64, 64, 32), dtype=np.int8)
        }
        d = r(d)

        for k, v in d.items():
            if isinstance(v, np.ndarray):
                print(k, v.shape)
            else:
                print(k, v)

    def test_rotate_tensor(self):
        r = Rotated(('image', 'label'), [0.0, 1.0, 0.0])

        d = {
            'image': torch.zeros((1, 64, 64, 32), device="cpu", dtype=torch.float32),
            'label': torch.ones((1, 64, 64, 32), device="cpu", dtype=torch.int8)
        }
        d = r(d)

        for k, v in d.items():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                print(k, v.shape)
            else:
                print(k, v)

    def test_rotate_apply(self):
        c = Compose([
            Rotated(('image', 'label'), (0.0, 3.14159265 / 2, 0.0)),
            Applyd(('image', 'label'),
                   modes=(GridSampleMode.BILINEAR, GridSampleMode.NEAREST),
                   padding_modes=(GridSamplePadMode.BORDER, GridSamplePadMode.BORDER))
        ])

        image = torch.zeros((1, 16, 16, 4), device="cpu", dtype=torch.float32)
        for y in range(image.shape[-2]):
            for z in range(image.shape[-1]):
                image[0, :, y, z] = y + z * 16
        label = torch.ones((1, 16, 16, 4), device="cpu", dtype=torch.int8)
        d = {
            'image': image,
            'label': label
        }
        # plt.imshow(d['image'][0, ..., d['image'].shape[-1]//2])
        d = c(d)
        # plt.imshow(d['image'][0, ..., d['image'].shape[-1]//2])
        print(d['image'].shape)

    def test_old_affine(self):
        c = Compose([
            Affined(('image', 'label'),
                    rotate_params=(0.0, 0.0, 3.14159265 / 2))
        ])

        d = {
            'image': torch.zeros((1, 64, 64, 32), device="cpu", dtype=torch.float32),
            'label': torch.ones((1, 64, 64, 32), device="cpu", dtype=torch.int8)
        }
        d = c(d)
        print(d['image'].shape)


class TestUtils(unittest.TestCase):

    def test_value_to_tuple_range(self):
        self.assertTupleEqual(value_to_tuple_range(5), (-5, 5))
        self.assertTupleEqual(value_to_tuple_range([5]), (-5, 5))
        self.assertTupleEqual(value_to_tuple_range((5,)), (-5, 5))
        self.assertTupleEqual(value_to_tuple_range([-2.1, 4.3]), (-2.1, 4.3))
        self.assertTupleEqual(value_to_tuple_range((-2.1, 4.3)), (-2.1, 4.3))
        self.assertTupleEqual(value_to_tuple_range([4.3, -2.1]), (-2.1, 4.3))
        self.assertTupleEqual(value_to_tuple_range((4.3, -2.1)), (-2.1, 4.3))


# Utility transforms for compose compiler
# =================================================================================================

class TestMemoryCacheMechanism(CacheMechanism):

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
                             cache=TestMemoryCacheMechanism(4))

        first = ct("foo")
        second = ct("foo")
        third = ct("bar")

        self.assertIs(first, second)
        self.assertIsNot(first, third)

    def test_multi_transform(self):

        def fake_multi_sample(keys, num_samples, roi_size):
            def _inner(t):
                for i in range(num_samples):
                    yield {'image': t[i:i+roi_size[0], i:i+roi_size[1]]}
            return _inner

#        t1 = RandSpatialCropSamplesd(keys=('image',), num_samples=4, roi_size=(32, 32))
        t1 = fake_multi_sample(keys=('image',), num_samples=4, roi_size=(32, 32))
        t2 = RandRotated(keys=('image',), range_z=(-torch.pi/2, torch.pi/2))
        c = Compose([t1, t2])

        d = torch.rand((1, 64, 64))

        _d = d.data
        _dd = d.data.clone()
        d.data = _dd
        r = c({'image': d})
