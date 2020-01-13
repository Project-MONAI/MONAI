
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


import monai
from monai.utils.aliases import alias
from monai.utils.mathutils import zipWith
from monai.utils.decorators import RestartGenerator
from functools import wraps
import numpy as np

export = monai.utils.export("monai.data.streams")


@export
@alias("ordertype")
class OrderType(object):
    SHUFFLE = "shuffle"
    CHOICE = "choice"
    LINEAR = "linear"


@export
@alias("datastream")
class DataStream(object):
    """
    The DataStream class represents a chain of iterable objects where one iterates over its source and in turn yields 
    values which are possibly transformed. This allows an intermediate object in the stream to modify a data element 
    which passes through the stream or generate more than one output value for each input. A sequence of stream objects 
    is created by using one stream as the source to another.
    
    This relies on an input source which must be an iterable. Values are taken from this in order and then passed to the 
    generate() generator method to produce one or more items, which are then yielded. Subclasses can override generate() 
    to produce filter or transformer types to place in a sequence of DataStream objects. The `streamgen` decorator can 
    be used to do the same. 
    
    Internal infrastructure can be setup when the iteration starts and can rely on the self.isRunning to indicate when
    generation is expected. When this changes to False methods are expected to cleanup and exit gracefully, and be able
    to be called again with isRunning set back to True. This allows restarting a complex stream object which may use
    threads requiring starting and stopping. The stop() method when called set isRunning to False and attempts to call
    the same on self.src, this is meant to be used to stop any internal processes (ie. threads) when iteration stops
    with the expectation that it can be restarted later. Reading isRunning or assigning a literal value to it is atomic
    thus thread-safe but keep this in mind when assigning a compound expression.
    """

    def __init__(self, src):
        """Initialize with `src' as the source iterable, and self.isRunning as True."""
        self.src = src
        self.isRunning = True

    def __iter__(self):
        """
        Iterate over every value from self.src, passing through self.generate() and yielding the
        values it generates.
        """
        self.isRunning = True
        for srcVal in self.src:
            for outVal in self.generate(srcVal):
                yield outVal  # yield with syntax too new?

    def generate(self, val):
        """Generate values from input `val`, by default just yields that. """
        yield val

    def stop(self):
        """Sets self.isRunning to False and calls stop() on self.src if it has this method."""
        self.isRunning = False
        if callable(getattr(self.src, "stop", None)):
            self.src.stop()

    def getGenFunc(self):
        """Returns a callable taking no arguments which produces the next item in the stream whenever called."""
        stream = iter(self)
        return lambda: next(stream)


class FuncStream(DataStream):
    """For use with `streamgen`, the given callable is used as the generator in place of generate()."""

    def __init__(self, src, func, fargs, fkwargs):
        super().__init__(src)
        self.func = func
        self.fargs = fargs
        self.fkwargs = fkwargs

    def generate(self, val):
        for outVal in self.func(val, *self.fargs, **self.fkwargs):
            yield outVal


@export
def streamgen(func):
    """
    Converts a generator function into a constructor for creating FuncStream instances 
    using the function as the generator.
    """

    @wraps(func)
    def _wrapper(src, *args, **kwargs):
        return FuncStream(src, func, args, kwargs)

    return _wrapper
        

@export
@alias("cachestream")
class CacheStream(DataStream):
    """
    Reads a finite number of items from the source, or everything, into a cache then yields them either in
    order, shuffled, or by choice indefinitely.
    """

    def __init__(self, src, bufferSize=None, orderType=OrderType.LINEAR):
        super().__init__(src)
        self.bufferSize = bufferSize
        self.orderType = orderType
        self.buffer = []
        
    def __iter__(self):
        self.buffer=[item for i, item in enumerate(self.src) if self.bufferSize is None or i<self.bufferSize]

        while self.isRunning:
            inds=np.arange(0,len(self.buffer))
            
            if self.orderType == OrderType.SHUFFLE:
                np.random.shuffle(inds)
            elif self.orderType == OrderType.CHOICE:
                inds=np.random.choice(inds,len(self.buffer))
            
            for i in inds:
                for outVal in self.generate(self.buffer[i]):
                    yield outVal
        

@export
@alias("bufferstream")
class BufferStream(DataStream):
    """
    Accumulates a buffer of generated items, starting to yield them only when the buffer is filled and doing so until the
    buffer is empty. The buffer is filled by generate() which calls bufferFull() when full to allow subclasses to react.
    After this the buffer contents are yielded in order until the buffer is empty, then the filling process restarts.
    """

    def __init__(self, src, bufferSize=10, orderType=OrderType.LINEAR):
        super().__init__(src)
        self.bufferSize = bufferSize
        self.orderType = orderType
        self.buffer = []

    def bufferFull(self):
        """Called when the buffer is full and before emptying it."""

    def generate(self, val):
        if len(self.buffer) == self.bufferSize:
            self.bufferFull()  # call overridable callback to trigger action when buffer full
            
            if self.orderType == OrderType.SHUFFLE:
                np.random.shuffle(self.buffer)
            elif self.orderType == OrderType.CHOICE:
                inds=np.random.choice(np.arange(len(self.buffer)),len(self.buffer))
                self.buffer=[self.buffer[i] for i in inds]

            while len(self.buffer) > 0:
                yield self.buffer.pop(0)

        self.buffer.append(val)


@export
@alias("batchstream")
class BatchStream(BufferStream):
    """Collects values from the source together into a batch of the stated size, ie. stacks buffered items."""

    def __init__(self, src, batchSize, sendShortBatch=False, orderType=OrderType.LINEAR):
        super().__init__(src, batchSize, orderType)
        self.sendShortBatch = sendShortBatch

    def bufferFull(self):
        """Replaces the buffer's contents with the arrays stacked together into a single item."""
        if isinstance(self.buffer[0], np.ndarray):
            # stack all the arrays together
            batch = np.stack(self.buffer)
        else:
            # stack the arrays from each item into one
            batch = tuple(zipWith(np.stack, *self.buffer))

        self.buffer[:] = [batch]  # yield only the one item when emptying the buffer

    def __iter__(self):
        for srcVal in super().__iter__():
            yield srcVal

        # only true if the iteration has completed but items are left to make up a shortened batch
        if len(self.buffer) > 0 and self.sendShortBatch:
            self.bufferFull()
            yield self.buffer.pop()


@export
@alias("mergestream")
class MergeStream(DataStream):
    """Merge data from multiple iterators into generated tuples."""

    def __init__(self, *srcs):
        self.srcs = srcs
        super().__init__(RestartGenerator(self.yieldMergedValues))

    def yieldMergedValues(self):
        iters = [iter(s) for s in self.srcs]
        canContinue = True

        while self.isRunning and canContinue:
            try:
                values = []
                for it in iters:
                    val = next(it)  # raises StopIteration when a source runs out of data at which point we quit

                    if not isinstance(val, (list, tuple)):
                        val = (val,)

                    values.append(tuple(val))

                srcVal = sum(values, ())

                for outVal in self.generate(srcVal):
                    yield outVal
            # must be caught as StopIteration won't propagate but magically mutate into RuntimeError
            except StopIteration:
                canContinue = False


@export
@alias("cyclingstream")
class CyclingStream(DataStream):
    def __init__(self, *srcs):
        self.srcs = srcs
        super().__init__(RestartGenerator(self.yieldAlternatingValues))

    def yieldAlternatingValues(self):
        iters = [iter(s) for s in self.srcs]
        canContinue = True

        while self.isRunning and canContinue:
            try:
                for it in iters:
                    srcVal = next(it)  # raises StopIteration when a source runs out of data at which point we quit
                    for outVal in self.generate(srcVal):
                        yield outVal

            # must be caught as StopIteration won't propagate but magically mutate into RuntimeError
            except StopIteration:
                canContinue = False


@export
class PrefetchStream(DataStream):
    """
    Calculates item dtype and shape before iteration. This will get a value from `src` in the constructor, assign it to
    self.srcVal, then assign the dtypes and shapes of the arrays to self.dtypes and self.shapes respectively. When it is 
    iterated over self.srcVal is yielded first followed by whatever else `src` produces so no data is lost.
    """

    def __init__(self, src):
        self.origSrc = src
        self.it = iter(src)
        self.srcVal = next(self.it)

        if isinstance(self.srcVal, np.ndarray):
            self.dtypes = self.srcVal.dtype
            self.shapes = self.srcVal.shape
        else:
            self.dtypes = tuple(b.dtype for b in self.srcVal)
            self.shapes = tuple(b.shape for b in self.srcVal)

        super().__init__(RestartGenerator(self._getSrc))

    def _getSrc(self):
        if self.it is not None:
            yield self.srcVal
        else:
            self.it = iter(self.origSrc)  # self.it is None when restarting so recreate the iterator here

        for srcVal in self.it:
            yield srcVal

        self.it = None

        
@export
@alias("finitestream")       
class FiniteStream(DataStream):
    """Yields only the specified number of items before quiting."""
    def __init__(self, src, numItems):
        super().__init__(src)
        self.numItems = numItems
        
    def __iter__(self):
        for _, item in zip(range(self.numItems), super().__iter__()):
            yield item
            

@export
@alias("tracestream")
class TraceStream(DataStream):
    def generate(self, val):
        vals = val if isinstance(val, (tuple, list)) else (val,)

        sizes = ", ".join("%s%s" % (s.dtype, s.shape) for s in vals)

        print("Stream -> %s" % sizes, flush=True)

        yield val
