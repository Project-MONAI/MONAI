import monai
from monai.utils.aliases import alias
from monai.data.streams import DataStream
from queue import Queue, Full, Empty
from threading import Thread


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

    def __init__(self, src, bufferSize=1, timeout=0.01):
        super().__init__(src)
        self.bufferSize = bufferSize
        self.timeout = timeout
        self.buffer = Queue(self.bufferSize)
        self.genThread = None

    def enqueueValues(self):
        # allows generate() to be overridden and used here (instead of iter(self.src))
        for srcVal in super().__iter__():
            while self.isRunning:
                try:
                    self.buffer.put(srcVal, timeout=self.timeout)
                except Full:
                    pass  # try to add the item again
                else:
                    break  # successfully added the item, quit trying
            else:  # quit the thread cleanly when requested to stop
                break

    def stop(self):
        super().stop()
        if self.genThread is not None:
            self.genThread.join()

    def __iter__(self):
        self.genThread = Thread(target=self.enqueueValues, daemon=True)
        self.genThread.start()
        self.isRunning = True

        try:
            while self.isRunning and (self.genThread.is_alive() or not self.buffer.empty()):
                try:
                    yield self.buffer.get(timeout=self.timeout)
                except Empty:
                    pass  # queue was empty this time, try again
        finally:
            self.stop()
