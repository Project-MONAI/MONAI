import unittest

import json
from io import StringIO

import torch

from monai.transforms.io.functional import load_geometry
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

if __name__ == '__main__':
    unittest.main()