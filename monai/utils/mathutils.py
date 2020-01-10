import itertools
import numpy as np


def zipWith(op, *vals, mapfunc=map):
    """
    Map `op`, using `mapfunc`, to each tuple derived from zipping the iterables in `vals'.
    """
    return mapfunc(op, zip(*vals))


def starZipWith(op, *vals):
    """
    Use starmap as the mapping function in zipWith.
    """
    return zipWith(op, *vals, mapfunc=itertools.starmap)


def first(iterable, default=None):
    """
    Returns the first item in the given iterable or `default` if empty, meaningful mostly with 'for' expressions.
    """
    for i in iterable:
        return i
    return default


def ensureTuple(vals):
    if not isinstance(vals, (list, tuple)):
        vals = (vals,)

    return tuple(vals)
