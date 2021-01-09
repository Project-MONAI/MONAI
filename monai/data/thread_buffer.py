# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from queue import Empty, Full, Queue
from threading import Thread


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

    def __init__(self, src, buffer_size=1, timeout=0.01):
        self.src = src
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.buffer = Queue(self.buffer_size)
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
