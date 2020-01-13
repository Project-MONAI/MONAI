from threading import Lock

import numpy as np

import monai
from monai.data.streams import DataStream, OrderType
from monai.utils.decorators import RestartGenerator


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

    def __init__(self, *arrays, order_type=OrderType.LINEAR, do_once=False, choice_probs=None):
        if order_type not in (OrderType.SHUFFLE, OrderType.CHOICE, OrderType.LINEAR):
            raise ValueError("Invalid order_type value %r" % (order_type,))

        self.arrays = ()
        self.order_type = order_type
        self.do_once = do_once
        self.choice_probs = None
        self.lock = Lock()

        super().__init__(RestartGenerator(self.yield_arrays))

        self.append_arrays(*arrays, choice_probs=choice_probs)

    def yield_arrays(self):
        while self.is_running:
            with self.lock:
                # capture locally so that emptying the reader doesn't interfere with an on-going interation
                arrays = self.arrays
                choice_probs = self.choice_probs

            indices = np.arange(arrays[0].shape[0] if arrays else 0)

            if self.order_type == OrderType.SHUFFLE:
                np.random.shuffle(indices)
            elif self.order_type == OrderType.CHOICE:
                indices = np.random.choice(indices, indices.shape, p=choice_probs)

            for i in indices:
                yield tuple(arr[i] for arr in arrays)

            if self.do_once or not arrays:  # stop first time through or if empty
                break

    def get_sub_arrays(self, indices):
        """Get a new ArrayReader with a subset of this one's data defined by the `indices` list."""
        with self.lock:
            sub_arrays = [a[indices] for a in self.arrays]
            sub_probs = None

            if self.choice_probs is not None:
                sub_probs = self.choice_probs[indices]
                sub_probs = sub_probs / np.sum(sub_probs)

        return ArrayReader(*sub_arrays, order_type=self.order_type, do_once=self.do_once, choice_probs=sub_probs)

    def append_arrays(self, *arrays, choice_probs=None):
        """
        Append the given arrays to the existing entries in self.arrays, or replacing self.arrays if this is empty. If
        `choice_probs` is provided this is appended to self.choice_probs, or replaces it if the latter is None or empty.
        """
        array_len = arrays[0].shape[0] if arrays else 0

        if array_len > 0 and any(arr.shape[0] != array_len for arr in arrays):
            raise ValueError("All input arrays must have the same length for dimension 0")

        with self.lock:
            if not self.arrays and arrays:
                self.arrays = tuple(arrays)
            elif array_len > 0:
                self.arrays = tuple(np.concatenate(ht) for ht in zip(self.arrays, arrays))

            if self.arrays and choice_probs is not None and choice_probs.shape[0] > 0:
                choice_probs = np.atleast_1d(choice_probs)

                if choice_probs.shape[0] != array_len:
                    raise ValueError("Length of choice_probs (%i) must match that of input arrays (%i)" %
                                     (self.choice_probs.shape[0], array_len))

                if self.choice_probs is None:
                    self.choice_probs = choice_probs
                else:
                    self.choice_probs = np.concatenate([self.choice_probs, choice_probs])

                self.choice_probs = self.choice_probs / np.sum(self.choice_probs)

    def empty_arrays(self):
        """Clear the stored arrays and choice_probs so that this reader is empty but functional."""
        with self.lock:
            self.arrays = ()
            self.choice_probs = None if self.choice_probs is None else self.choice_probs[:0]

    def __len__(self):
        return len(self.arrays[0]) if self.arrays else 0
