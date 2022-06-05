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


from multiprocessing.context import SpawnContext
from queue import Empty, Full, Queue
from threading import Thread

import torch

from monai.data import DataLoader, Dataset


class ThreadBuffer:
    """
    Iterates over values from self.src in a separate thread but yielding them in the current thread. This allows values
    to be queued up asynchronously. The internal thread will continue running so long as the source has values or until
    the stop() method is called.

    One issue raised by using a thread in this way is that during the lifetime of the thread the source object is being
    iterated over, so if the thread hasn't finished another attempt to iterate over it will raise an exception or yield
    unexpected results. To ensure the thread releases the iteration and proper cleanup is done the stop() method must
    be called which will join with the thread.

    Args:
        src: Source data iterable
        buffer_size: Number of items to buffer from the source
        timeout: Time to wait for an item from the buffer, or to wait while the buffer is full when adding items
    """

    def __init__(self, src, buffer_size: int = 1, timeout: float = 0.01):
        self.src = src
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.buffer: Queue = Queue(self.buffer_size)
        self.gen_thread = None
        self.is_running = False

    def enqueue_values(self):
        for src_val in self.src:
            while self.is_running:
                try:
                    self.buffer.put(src_val, timeout=self.timeout)
                except Full:
                    pass  # try to add the item again
                else:
                    break  # successfully added the item, quit trying
            else:  # quit the thread cleanly when requested to stop
                break

    def stop(self):
        self.is_running = False  # signal the thread to exit

        if self.gen_thread is not None:
            self.gen_thread.join()

        self.gen_thread = None

    def __iter__(self):

        self.is_running = True
        self.gen_thread = Thread(target=self.enqueue_values, daemon=True)
        self.gen_thread.start()

        try:
            while self.is_running and (self.gen_thread.is_alive() or not self.buffer.empty()):
                try:
                    yield self.buffer.get(timeout=self.timeout)
                except Empty:
                    pass  # queue was empty this time, try again
        finally:
            self.stop()  # ensure thread completion


def buffer_iterator(src, buffer_size: int = 1, timeout: float = 0.01, repeats:int = 1):
    """
    Create a ThreadBuffer object using the `src`, `buffer_size`, and `timeout` parameters given for the constructor 
    aguments of the same names, and yield each generated object `repeats` number of times successively.
    
    Args:
        src: Source data iterable
        buffer_size: Number of items to buffer from the source
        timeout: Time to wait for an item from the buffer, or to wait while the buffer is full when adding items
        repeats: Number of repeat generations to perform which is asynchronous from the generation of the next value
        
    Returns:
        Generator yield (repeated) values from `src` asynchronously
    """
    buffer = ThreadBuffer(src=src, buffer_size=buffer_size, timeout=timeout)

    for batch in buffer:
        for _ in range(repeats):
            yield batch
            

class _ProcessThread(Thread):
    """Shim class to make a thread look like a process to the DataLoader class."""

    @property
    def pid(self):
        return id(self)

    def run(self):
        try:
            super().run()
        finally:
            torch.utils.data._utils.worker._worker_info = None  # clean up global data used for processes


class _ProcessQueue(Queue):
    """Shim class to make a thread queue look like a process queue to the DataLoader class."""

    def close(self):
        pass

    def cancel_join_thread(self):
        pass


class _ProcessThreadContext(SpawnContext):
    _name = "processthread"

    # threads will be created which looks like processes
    Process = _ProcessThread  # type: ignore
    # thread queue used in place of process queue to avoid some weird cleanup errors
    Queue = _ProcessQueue  # type: ignore


class ThreadDataLoader(DataLoader):
    """
    Subclass of `DataLoader` using a `ThreadBuffer` object to implement `__iter__` method asynchronously. This will
    iterate over data from the loader as expected however the data is generated on a separate thread. Use this class
    where a `DataLoader` instance is required and not just an iterable object.

    The default behaviour with `repeats` set to 1 is to yield each batch as it is generated, however with a higher
    value the generated batch is yielded that many times while underlying dataset asynchronously generates the next.
    Typically not all relevant information is learned from a batch in a single iteration so training multiple times
    on the same batch will still produce good training with minimal short-term overfitting while allowing a slow batch
    generation process more time to produce a result. This duplication is done by simply yielding the same object many
    times and not by regenerating the data.

    Another typical usage is to accelerate light-weight preprocessing (usually cached all the deterministic transforms
    and no IO operations), because it leverages the separate thread to execute preprocessing to avoid unnecessary IPC
    between multiple workers of DataLoader. And as CUDA may not work well with the multi-processing of DataLoader,
    `ThreadDataLoader` can be useful for GPU transforms. For more details:
    https://github.com/Project-MONAI/tutorials/blob/master/acceleration/fast_model_training_guide.md.

    The `use_thread_workers` will cause workers to be created as threads rather than processes although everything else
    in terms of how the class works is unchanged. This allows multiple workers to be used in Windows for example, or in
    any other situation where thread semantics is desired.

    See:
        * Fischetti et al. "Faster SGD training by minibatch persistency." ArXiv (2018) https://arxiv.org/abs/1806.07353
        * Dami et al., "Faster Neural Network Training with Data Echoing" ArXiv (2020) https://arxiv.org/abs/1907.05550
        * Ramezani et al. "GCN meets GPU: Decoupling "When to Sample" from "How to Sample"." NeurIPS (2020).
          https://proceedings.neurips.cc/paper/2020/file/d714d2c5a796d5814c565d78dd16188d-Paper.pdf

    Args:
        dataset: input dataset.
        buffer_size: number of items to buffer from the data source.
        buffer_timeout: time to wait for an item from the buffer, or to wait while the buffer is full when adding items.
        repeats: number of times to yield the same batch.
        use_thread_workers: if True and num_workers > 0 the workers are created as threads instead of processes
        kwargs: other arguments for `DataLoader` except for `dataset`.

    """

    def __init__(
        self,
        dataset: Dataset,
        buffer_size: int = 1,
        buffer_timeout: float = 0.01,
        repeats: int = 1,
        use_thread_workers: bool = False,
        **kwargs,
    ):
        # if workers should be threads, create a new multiprocessing context with the process and queue types
        # substituted with the shim types given above
        if use_thread_workers and kwargs.get("num_workers", 0) > 0:
            kwargs["multiprocessing_context"] = _ProcessThreadContext()
            kwargs["persistent_workers"] = False

        super().__init__(dataset, **kwargs)
        self.buffer_size = buffer_size
        self.buffer_timeout = buffer_timeout
        self.repeats = repeats

    def __iter__(self):
        yield from buffer_iterator(super().__iter__(), self.buffer_size, self.buffer_timeout, self.repeats)
