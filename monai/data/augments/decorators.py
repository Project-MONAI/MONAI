
# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import wraps
from monai.utils.arrayutils import randChoice, zeroMargins
import numpy as np


def augment(prob=0.5, applyIndices=None):
    """
    Creates an augmentation function when decorating to a function returning an array-modifying callable. The function
    this decorates is given the list of input arrays as positional arguments and then should return a callable operation
    which performs the augmentation. This wrapper then chooses whether to apply the operation to the arguments and if so
    to which ones. The `prob' argument states the probability the augment is applied, `applyIndices' gives indices of
    the arrays to apply to (or None for all). The arguments are also keyword arguments in the resulting function.
    """

    def _inner(func):
        @wraps(func)
        def _func(*args, **kwargs):
            _prob = kwargs.pop("prob", prob)  # get the probability of applying this augment

            if _prob < 1.0 and not randChoice(_prob):  # if not chosen just return the original argument
                return args

            _applyIndices = kwargs.pop("applyIndices", applyIndices)

            op = func(*args, **kwargs)
            indices = list(_applyIndices or range(len(args)))

            return tuple((op(im) if i in indices else im) for i, im in enumerate(args))

        if _func.__doc__:
            _func.__doc__ += """
       
Added keyword arguments:
    prob: probability of applying this augment (default: 0.5)
    applyIndices: indices of arrays to apply augment to (default: None meaning all)
"""
        return _func

    return _inner


def checkSegmentMargin(func):
    """
    Decorate an augment callable `func` with a check to ensure a given segmentation image in the set does not
    touch the margins of the image when geometric transformations are applied. The keyword arguments `margin`,
    `maxCount` and `nonzeroIndex` are used to check the image at index `nonzeroIndex` has the given margin of
    pixels around its edges, trying `maxCount` number of times to get a modifier by calling `func` before 
    giving up and producing a identity modifier in its place. 
    """

    @wraps(func)
    def _check(*args, **kwargs):
        margin = max(1, kwargs.pop("margin", 5))
        maxCount = max(1, kwargs.pop("maxCount", 5))
        nonzeroIndex = kwargs.pop("nonzeroIndex", -1)
        acceptedOutput = False

        while maxCount > 0 and not acceptedOutput:
            op = func(*args, **kwargs)
            maxCount -= 1

            if nonzeroIndex == -1:
                acceptedOutput = True
            else:
                seg = op(args[nonzeroIndex]).astype(np.int32)
                acceptedOutput = zeroMargins(seg, margin)

        if not acceptedOutput:
            op = lambda arr: arr

        return op

    return _check
