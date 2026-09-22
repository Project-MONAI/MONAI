import unittest
import torch
from parameterized import parameterized

from monai.networks.nets import SegResNetDS


class TestSegResNetDSShapeLogic(unittest.TestCase):
    """Tests for shape_factor() and is_valid_shape() in SegResNetDS."""

    # ---- shape_factor, isotropic (resolution=None) ----
    @parameterized.expand([
        # (spatial_dims, blocks_down, expected_factor)
        (2, [1, 2, 2, 4], [8, 8]),
        (3, [1, 2, 2, 4], [8, 8, 8]),
        (3, [1, 2, 4],    [4, 4, 4]),
    ])
    def test_shape_factor_isotropic(self, spatial_dims, blocks_down, expected):
        """
        Test shape_factor() calculation for isotropic (resolution=None) configurations.

        Args:
            spatial_dims: Number of spatial dimensions (2 or 3).
            blocks_down: List of integers defining the downsampling blocks.
            expected: Expected divisor factors per spatial dimension.
        """
        model = SegResNetDS(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=1,
            blocks_down=blocks_down,
            resolution=None,
        )
        actual = [int(x) for x in model.shape_factor()]
        self.assertEqual(actual, expected)

    # ---- shape_factor, anisotropic (resolution set) ----
    @parameterized.expand([
        # (spatial_dims, blocks_down, resolution, expected_factor)
        (3, [1, 2, 2, 4], [1, 1, 5], [8, 8, 2]),
        (3, [1, 2, 2, 4], [1, 2, 3], [8, 4, 4]),
    ])
    def test_shape_factor_anisotropic(self, spatial_dims, blocks_down, resolution, expected):
        """
        Test shape_factor() calculation for anisotropic (resolution set) configurations.

        Args:
            spatial_dims: Number of spatial dimensions.
            blocks_down: List of integers defining the downsampling blocks.
            resolution: List of resolutions for anisotropic scaling.
            expected: Expected divisor factors per spatial dimension.
        """
        model = SegResNetDS(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=1,
            blocks_down=blocks_down,
            resolution=resolution,
        )
        actual = [int(x) for x in model.shape_factor()]
        self.assertEqual(actual, expected)

    # ---- is_valid_shape, valid inputs ----
    @parameterized.expand([
        # (spatial_dims, blocks_down, resolution, input_shape)
        (2, [1, 2, 2, 4], None,       (1, 1, 16, 16)),
        (3, [1, 2, 2, 4], None,       (1, 1, 16, 16, 16)),
        (3, [1, 2, 2, 4], [1, 1, 5],  (1, 1, 16, 16, 16)),
        (3, [1, 2, 2, 4], [1, 2, 3],  (1, 1, 16, 16, 16)),
    ])
    def test_is_valid_shape_true(self, spatial_dims, blocks_down, resolution, shape):
        """
        Test is_valid_shape() returns True for inputs with valid shapes.

        Args:
            spatial_dims: Number of spatial dimensions.
            blocks_down: List of integers defining the downsampling blocks.
            resolution: List of resolutions for anisotropic scaling.
            shape: Input tensor shape to validate.
        """
        model = SegResNetDS(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=1,
            blocks_down=blocks_down,
            resolution=resolution,
        )
        x = torch.zeros(shape)
        self.assertTrue(model.is_valid_shape(x))

    # ---- is_valid_shape, invalid inputs ----
    @parameterized.expand([
        (3, [1, 2, 2, 4], None,       (1, 1, 15, 16, 16)),      # 15 not divisible by 8
        (3, [1, 2, 2, 4], None,       (1, 1, 7, 7, 7)),         # 7 not divisible by 8
        (3, [1, 2, 2, 4], [1, 1, 5],  (1, 1, 16, 15, 16)),      # 15 not divisible by 8
        (3, [1, 2, 2, 4], [1, 2, 3],  (1, 1, 16, 16, 15)),      # 15 not divisible by 4
    ])
    def test_is_valid_shape_false(self, spatial_dims, blocks_down, resolution, shape):
        """
        Test is_valid_shape() returns False for inputs with invalid shapes.

        Args:
            spatial_dims: Number of spatial dimensions.
            blocks_down: List of integers defining the downsampling blocks.
            resolution: List of resolutions for anisotropic scaling.
            shape: Input tensor shape to validate.
        """
        model = SegResNetDS(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=1,
            blocks_down=blocks_down,
            resolution=resolution,
        )
        x = torch.zeros(shape)
        self.assertFalse(model.is_valid_shape(x))

    # ---- integration: forward pass raises on invalid shape ----
    def test_forward_raises_on_invalid_shape(self):
        """
        Test that the forward pass raises ValueError when the input shape is invalid.
        """
        model = SegResNetDS(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            blocks_down=[1, 2, 2, 4],
            resolution=None,
        )
        x = torch.zeros(1, 1, 15, 16, 16)  # 15 not divisible by 8
        with self.assertRaises(ValueError):
            model(x)


if __name__ == "__main__":
    unittest.main()