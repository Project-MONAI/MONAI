from queue import Empty, Full, Queue
from threading import Thread

import monai
from monai.data.streams import DataStream
from monai.utils.aliases import alias


@monai.utils.export("monai.data.streams")
@alias("threadbufferstream")
class ThreadBufferStream(DataStream):
    """
    Iterates over values from self.src in a separate thread but yielding them in the current thread. This allows values
    to be queued up asynchronously. The internal thread will continue running so long as the source has values or until
    the stop() method is called.

    One issue raised by using a thread in this way is that during the lifetime of the thread the source object is being
    iterated over, so if the thread hasn't finished another attempt to iterate over it will raise an exception or yield
    inexpected results. To ensure the thread releases the iteration and proper cleanup is done the stop() method must
    be called which will join with the thread.
    """

    def __init__(self, src, buffer_size=1, timeout=0.01):
        super().__init__(src)
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.buffer = Queue(self.buffer_size)
        self.gen_thread = None

    def enqueue_values(self):
        # allows generate() to be overridden and used here (instead of iter(self.src))
        for src_val in super().__iter__():
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
        super().stop()
        if self.gen_thread is not None:
            self.gen_thread.join()

    def __iter__(self):
        self.gen_thread = Thread(target=self.enqueue_values, daemon=True)
        self.gen_thread.start()
        self.is_running = True

        try:
            while self.is_running and (self.gen_thread.is_alive() or not self.buffer.empty()):
                try:
                    yield self.buffer.get(timeout=self.timeout)
                except Empty:
                    pass  # queue was empty this time, try again
        finally:
            self.stop()
