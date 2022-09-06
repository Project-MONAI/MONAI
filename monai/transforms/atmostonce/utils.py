from typing import Union

import numpy as np

import torch

from monai.config import NdarrayOrTensor
from monai.utils.mapping_stack import Matrix, MetaMatrix


def matmul(
        first: Union[MetaMatrix, Matrix, NdarrayOrTensor],
        second: Union[MetaMatrix, Matrix, NdarrayOrTensor]
):
    matrix_types = (MetaMatrix, Matrix, torch.Tensor, np.ndarray)

    if not isinstance(first, matrix_types):
        raise TypeError(f"'first' must be one of {matrix_types} but is {type(first)}")
    if not isinstance(second, matrix_types):
        raise TypeError(f"'second' must be one of {matrix_types} but is {type(second)}")

    first_ = first
    if isinstance(first_, MetaMatrix):
        first_ = first_.matrix.matrix
    elif isinstance(first_, Matrix):
        first_ = first_.matrix

    second_ = second
    if isinstance(second_, MetaMatrix):
        second_ = second_.matrix.matrix
    elif isinstance(second_, Matrix):
        second_ = second_.matrix


    if isinstance(first_, np.ndarray):
        if isinstance(second_, np.ndarray):
            return first_ @ second_
        else:
            return torch.from_numpy(first_) @ second_
    else:
        if isinstance(second_, np.ndarray):
            return first_ @ torch.from_numpy(second_)
        else:
            return first_ @ second_
