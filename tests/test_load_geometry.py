import unittest

import json
from io import StringIO

import torch

from monai.data.meta_tensor import MetaTensor
from monai.transforms.io.functional import load_geometry, save_geometry
from monai.utils.enums import KindKeys


class TestLoadGeometry(unittest.TestCase):

    def test_load_geometry_2d(self):

        entry = {
            "schema": {
                "geometry": "point"
            },
            "points": [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1]
            ]
        }

        expected = torch.tensor(
            [[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]],
            dtype=torch.float32
        )
        file = StringIO(json.dumps(entry))

        points = load_geometry(file, None, None)

        self.assertEqual(points.shape, (4, 3))

        self.assertEqual(points.kind, KindKeys.POINT)

        self.assertEqual(points.dtype, torch.float32)

        self.assertTrue(torch.allclose(points.data, expected))


    def test_load_geometry_3d(self):

        entry = {
            "schema": {
                "geometry": "point"
            },
            "points": [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
            ]
        }

        expected = torch.tensor(
            [[0., 0., 0., 1.], [0., 1., 0., 1.], [0., 0., 1., 1.], [0., 1., 1., 1.],
             [1., 0., 0., 1.], [1., 1., 0., 1.], [1., 0., 1., 1.], [1., 1., 1., 1.]],
            dtype=torch.float32
        )
        file = StringIO(json.dumps(entry))

        points = load_geometry(file, None, None)

        self.assertEqual(points.shape, (8, 4))

        self.assertEqual(points.kind, KindKeys.POINT)

        self.assertEqual(points.dtype, torch.float32)

        self.assertTrue(torch.allclose(points.data, expected))


class TestSaveGeometry(unittest.TestCase):

    def test_save_geometry_2d(self):

        entry = {
            "schema": {
                "geometry": "point"
            },
            "points": [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0]
            ]
        }

        padded_points = [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        data = torch.as_tensor(padded_points, dtype=torch.float32)
        data = MetaTensor(data)
        data.kind = KindKeys.POINT

        # expected = torch.tensor(
        #     [[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]],
        #     dtype=torch.float32
        # )
        file = StringIO()

        save_geometry(data, file, None, None)
        actual = file.getvalue()
        expected = json.dumps(entry)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
