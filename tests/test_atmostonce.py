import unittest

import math

import numpy as np


import torch

from monai.transforms.atmostonce import array as amoa
from monai.transforms.atmostonce.lazy_transform import compile_transforms
from monai.utils import TransformBackends

from monai.transforms import Affined, Affine
from monai.transforms.atmostonce.functional import croppad, resize, rotate, spacing
from monai.transforms.atmostonce.apply import Applyd, extents_from_shape, shape_from_extents
from monai.transforms.atmostonce.dictionary import Rotated
from monai.transforms.compose import Compose
from monai.utils.enums import GridSampleMode, GridSamplePadMode
from monai.utils.mapping_stack import MatrixFactory


def get_img(size):
  img = torch.zeros(size, dtype=torch.float32)
  if len(size) == 2:
    for j in range(size[0]):
      for i in range(size[1]):
        img[j, i] = i + j * size[1]
  else:
    for k in range(size[-1]):
      for j in range(size[-2]):
        img[..., j, k] = j + k * size[0]
  return np.expand_dims(img, 0)


def enumerate_results_of_op(results):
    if isinstance(results, dict):
        for k, v in results.items():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                print(k, v.shape, v[tuple(slice(0, 8) for _ in r.shape)])
            else:
                print(k, v)
    else:
        for ir, r in enumerate(results):
            if isinstance(r, (np.ndarray, torch.Tensor)):
                print(ir, r.shape, r[tuple(slice(0, 8) for _ in r.shape)])
            else:
                print(ir, r)


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
        result = compile_transforms(values)
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

    # TODO: turn into proper test
    def test_spacing(self):
        results = spacing(np.zeros((1, 24, 32), dtype=np.float32),
                          (0.5, 0.6),
                          (1.0, 1.0),
                          False,
                          "bilinear",
                          "border",
                          False)


    # TODO: turn into proper test
    def test_resize(self):
        results = resize(np.zeros((1, 24, 32), dtype=np.float32),
                         (40, 40),
                         "all",
                         "bilinear",
                         False)
        enumerate_results_of_op(results)

    # TODO: turn into proper test
    def test_rotate(self):
        results = rotate(np.zeros((1, 64, 64), dtype=np.float32),
                         torch.pi / 4,
                         True,
                         "bilinear",
                         "border")
        enumerate_results_of_op(results)

        results = rotate(np.zeros((1, 64, 64), dtype=np.float32),
                         torch.pi / 4,
                         False,
                         "bilinear",
                         "border")
        enumerate_results_of_op(results)

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

    def test_croppad(self):
        img = get_img((15, 15)).astype(int)
        results = croppad(img, (slice(4, 8), slice(3, 9)))
        enumerate_results_of_op(results)
        m = results[1].matrix.matrix
        print(m)
        result_size = results[2]['spatial_shape']
        a = Affine(affine=m,
                   padding_mode=GridSamplePadMode.ZEROS,
                   spatial_size=result_size)
        img_, _ = a(img)
        print(img_.numpy())


class TestArrayTransforms(unittest.TestCase):

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


class TestRotateEulerd(unittest.TestCase):

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
