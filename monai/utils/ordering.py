# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np

from monai.utils.enums import OrderingTransformations, OrderingType


class Ordering:
    """
    Ordering class that projects a 2D or 3D image into a 1D sequence. It also allows the image to be transformed with
    one of the following transformations:
    Reflection (see np.flip for more details).
    Transposition (see np.transpose for more details).
    90-degree rotation (see np.rot90 for more details).

    The transformations are applied in the order specified by the transformation_order parameter.

    Args:
        ordering_type: The ordering type. One of the following:
            - 'raster_scan': The image is projected into a 1D sequence by scanning the image from left to right and from
            top to bottom. Also called a row major ordering.
            - 's_curve': The image is projected into a 1D sequence by scanning the image in a circular snake like
            pattern from top left towards right gowing in a spiral towards the center.
            - random': The image is projected into a 1D sequence by randomly shuffling the image.
        spatial_dims: The number of spatial dimensions of the image.
        dimensions: The dimensions of the image.
        reflected_spatial_dims: A tuple of booleans indicating whether to reflect the image along each spatial dimension.
        transpositions_axes: A tuple of tuples indicating the axes to transpose the image along.
        rot90_axes: A tuple of tuples indicating the axes to rotate the image along.
        transformation_order: The order in which to apply the transformations.
    """

    def __init__(
        self,
        ordering_type: str,
        spatial_dims: int,
        dimensions: tuple[int, int, int] | tuple[int, int, int, int],
        reflected_spatial_dims: tuple[bool, bool] | None = None,
        transpositions_axes: tuple[tuple[int, int], ...] | tuple[tuple[int, int, int], ...] | None = None,
        rot90_axes: tuple[tuple[int, int], ...] | None = None,
        transformation_order: tuple[str, ...] = (
            OrderingTransformations.TRANSPOSE.value,
            OrderingTransformations.ROTATE_90.value,
            OrderingTransformations.REFLECT.value,
        ),
    ) -> None:
        super().__init__()
        self.ordering_type = ordering_type

        if self.ordering_type not in list(OrderingType):
            raise ValueError(
                f"ordering_type must be one of the following {list(OrderingType)}, but got {self.ordering_type}."
            )

        self.spatial_dims = spatial_dims
        self.dimensions = dimensions

        if len(dimensions) != self.spatial_dims + 1:
            raise ValueError(f"dimensions must be of length {self.spatial_dims + 1}, but got {len(dimensions)}.")

        self.reflected_spatial_dims = reflected_spatial_dims
        self.transpositions_axes = transpositions_axes
        self.rot90_axes = rot90_axes
        if len(set(transformation_order)) != len(transformation_order):
            raise ValueError(f"No duplicates are allowed. Received {transformation_order}.")

        for transformation in transformation_order:
            if transformation not in list(OrderingTransformations):
                raise ValueError(
                    f"Valid transformations are {list(OrderingTransformations)} but received {transformation}."
                )
        self.transformation_order = transformation_order

        self.template = self._create_template()
        self._sequence_ordering = self._create_ordering()
        self._revert_sequence_ordering = np.argsort(self._sequence_ordering)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x[self._sequence_ordering]

        return x

    def get_sequence_ordering(self) -> np.ndarray:
        return self._sequence_ordering

    def get_revert_sequence_ordering(self) -> np.ndarray:
        return self._revert_sequence_ordering

    def _create_ordering(self) -> np.ndarray:
        self.template = self._transform_template()
        order = self._order_template(template=self.template)

        return order

    def _create_template(self) -> np.ndarray:
        spatial_dimensions = self.dimensions[1:]
        template = np.arange(np.prod(spatial_dimensions)).reshape(*spatial_dimensions)

        return template

    def _transform_template(self) -> np.ndarray:
        for transformation in self.transformation_order:
            if transformation == OrderingTransformations.TRANSPOSE.value:
                self.template = self._transpose_template(template=self.template)
            elif transformation == OrderingTransformations.ROTATE_90.value:
                self.template = self._rot90_template(template=self.template)
            elif transformation == OrderingTransformations.REFLECT.value:
                self.template = self._flip_template(template=self.template)

        return self.template

    def _transpose_template(self, template: np.ndarray) -> np.ndarray:
        if self.transpositions_axes is not None:
            for axes in self.transpositions_axes:
                template = np.transpose(template, axes=axes)

        return template

    def _flip_template(self, template: np.ndarray) -> np.ndarray:
        if self.reflected_spatial_dims is not None:
            for axis, to_reflect in enumerate(self.reflected_spatial_dims):
                template = np.flip(template, axis=axis) if to_reflect else template

        return template

    def _rot90_template(self, template: np.ndarray) -> np.ndarray:
        if self.rot90_axes is not None:
            for axes in self.rot90_axes:
                template = np.rot90(template, axes=axes)

        return template

    def _order_template(self, template: np.ndarray) -> np.ndarray:
        depths = None
        if self.spatial_dims == 2:
            rows, columns = template.shape[0], template.shape[1]
        else:
            rows, columns, depths = (template.shape[0], template.shape[1], template.shape[2])

        sequence = eval(f"self.{self.ordering_type}_idx")(rows, columns, depths)

        ordering = np.array([template[tuple(e)] for e in sequence])

        return ordering

    @staticmethod
    def raster_scan_idx(rows: int, cols: int, depths: int | None = None) -> np.ndarray:
        idx: list[tuple] = []

        for r in range(rows):
            for c in range(cols):
                if depths is not None:
                    for d in range(depths):
                        idx.append((r, c, d))
                else:
                    idx.append((r, c))

        idx_np = np.array(idx)

        return idx_np

    @staticmethod
    def s_curve_idx(rows: int, cols: int, depths: int | None = None) -> np.ndarray:
        idx: list[tuple] = []

        for r in range(rows):
            col_idx = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            for c in col_idx:
                if depths:
                    depth_idx = range(depths) if c % 2 == 0 else range(depths - 1, -1, -1)

                    for d in depth_idx:
                        idx.append((r, c, d))
                else:
                    idx.append((r, c))

        idx_np = np.array(idx)

        return idx_np

    @staticmethod
    def random_idx(rows: int, cols: int, depths: int | None = None) -> np.ndarray:
        idx: list[tuple] = []

        for r in range(rows):
            for c in range(cols):
                if depths:
                    for d in range(depths):
                        idx.append((r, c, d))
                else:
                    idx.append((r, c))

        idx_np = np.array(idx)
        np.random.shuffle(idx_np)

        return idx_np
