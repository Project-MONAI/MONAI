import numpy as np


class ShapeFormat:
    """ShapeFormat defines meanings for the data in a MedicalImage.
    Image data is a numpy's ndarray. Without shape format, it is impossible to know what each
    dimension means.

    NOTE: ShapeFormat objects are immutable.

    """

    CHW = 'CHW'
    CHWD = 'CHWD'


def get_shape_format(img: np.ndarray):
    """Return the shape format of the image data

    Args:
        img (np.ndarray): the image data

    Returns: a shape format or None

    Raise: AssertionError if any of the specified args is invalid

    """
    assert isinstance(img, np.ndarray), 'invalid value img - must be np.ndarray'
    if img.ndim == 3:
        return ShapeFormat.CHW
    elif img.ndim == 4:
        return ShapeFormat.CHWD
    else:
        return None
