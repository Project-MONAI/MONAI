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

from multiprocessing.pool import ThreadPool

import numpy as np

from monai.data.streams.datastream import BatchStream, DataStream, OrderType


class AugmentStream(DataStream):
    """Applies the given augmentations in generate() to each given value and yields the results."""

    def __init__(self, src, augments=[]):
        super().__init__(src)
        self.augments = list(augments)

    def generate(self, val):
        yield self.apply_augments(val)

    def apply_augments(self, arrays):
        """Applies augments to the data tuple `arrays` and returns the result."""
        to_tuple = isinstance(arrays, np.ndarray)
        arrays = (arrays,) if to_tuple else arrays

        for aug in self.augments:
            arrays = aug(*arrays)

        return arrays[0] if to_tuple else arrays


class ThreadAugmentStream(BatchStream, AugmentStream):
    """
    Applies the given augmentations to each value from the source using multiple threads. Resulting batches are yielded
    synchronously so the client must wait for the threads to complete.
    """

    def __init__(self, src, batch_size, num_threads=None, augments=[], order_type=OrderType.LINEAR):
        BatchStream.__init__(self, src, batch_size, False, order_type)
        AugmentStream.__init__(self, src, augments)
        self.num_threads = num_threads
        self.pool = None

    def _augment_thread_func(self, index, arrays):
        self.buffer[index] = self.apply_augments(arrays)

    def apply_augments_threaded(self):
        self.pool.starmap(self._augment_thread_func, enumerate(self.buffer))

    def buffer_full(self):
        self.apply_augments_threaded()
        super().buffer_full()

    def __iter__(self):
        with ThreadPool(self.num_threads) as self.pool:
            for src_val in super().__iter__():
                yield src_val
