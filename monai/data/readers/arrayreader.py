from threading import Lock
import monai
from monai.utils.decorators import RestartGenerator
from monai.data.streams import DataStream, OrderType
import numpy as np


@monai.utils.export("monai.data.readers")
class ArrayReader(DataStream):
    """
    Creates a data source from one or more equal length arrays. Each data item yielded is a tuple of slices
    containing a single index in the 0th dimension (ie. batch dimension) for each array. By default values
    are drawn in sequential order but can be set to shuffle the order so that each value appears exactly once
    per epoch, or to choose a random selection which may include items multiple times or not at all based off
    an optional probability distribution. By default the stream will iterate over the arrays indefinitely or
    optionally only once.
    """

    def __init__(self, *arrays, orderType=OrderType.LINEAR, doOnce=False, choiceProbs=None):
        if orderType not in (OrderType.SHUFFLE, OrderType.CHOICE, OrderType.LINEAR):
            raise ValueError("Invalid orderType value %r" % (orderType,))

        self.arrays = ()
        self.orderType = orderType
        self.doOnce = doOnce
        self.choiceProbs = None
        self.lock = Lock()

        super().__init__(RestartGenerator(self.yieldArrays))

        self.appendArrays(*arrays, choiceProbs=choiceProbs)

    def yieldArrays(self):
        while self.isRunning:
            with self.lock:
                # capture locally so that emptying the reader doesn't interfere with an on-going interation
                arrays = self.arrays
                choiceProbs = self.choiceProbs

            indices = np.arange(arrays[0].shape[0] if arrays else 0)

            if self.orderType == OrderType.SHUFFLE:
                np.random.shuffle(indices)
            elif self.orderType == OrderType.CHOICE:
                indices = np.random.choice(indices, indices.shape, p=choiceProbs)

            for i in indices:
                yield tuple(arr[i] for arr in arrays)

            if self.doOnce or not arrays:  # stop first time through or if empty
                break

    def getSubArrays(self, indices):
        """Get a new ArrayReader with a subset of this one's data defined by the `indices` list."""
        with self.lock:
            subArrays = [a[indices] for a in self.arrays]
            subProbs = None

            if self.choiceProbs is not None:
                subProbs = self.choiceProbs[indices]
                subProbs = subProbs / np.sum(subProbs)

        return ArrayReader(*subArrays, orderType=self.orderType, doOnce=self.doOnce, choiceProbs=subProbs)

    def appendArrays(self, *arrays, choiceProbs=None):
        """
        Append the given arrays to the existing entries in self.arrays, or replacing self.arrays if this is empty. If
        `choiceProbs` is provided this is appended to self.choiceProbs, or replaces it if the latter is None or empty.
        """
        arrayLen = arrays[0].shape[0] if arrays else 0

        if arrayLen > 0 and any(arr.shape[0] != arrayLen for arr in arrays):
            raise ValueError("All input arrays must have the same length for dimension 0")

        with self.lock:
            if not self.arrays and arrays:
                self.arrays = tuple(arrays)
            elif arrayLen > 0:
                self.arrays = tuple(np.concatenate(ht) for ht in zip(self.arrays, arrays))

            if self.arrays and choiceProbs is not None and choiceProbs.shape[0] > 0:
                choiceProbs = np.atleast_1d(choiceProbs)

                if choiceProbs.shape[0] != arrayLen:
                    raise ValueError("Length of choiceProbs (%i) must match that of input arrays (%i)" % 
                                     (self.choiceProbs.shape[0], arrayLen))

                if self.choiceProbs is None:
                    self.choiceProbs = choiceProbs
                else:
                    self.choiceProbs = np.concatenate([self.choiceProbs, choiceProbs])

                self.choiceProbs = self.choiceProbs / np.sum(self.choiceProbs)

    def emptyArrays(self):
        """Clear the stored arrays and choiceProbs so that this reader is empty but functional."""
        with self.lock:
            self.arrays = ()
            self.choiceProbs = None if self.choiceProbs is None else self.choiceProbs[:0]

    def __len__(self):
        return len(self.arrays[0]) if self.arrays else 0
