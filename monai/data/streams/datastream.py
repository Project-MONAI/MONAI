from functools import wraps

import numpy as np

import monai
from monai.utils.aliases import alias
from monai.utils.decorators import RestartGenerator
from monai.utils.mathutils import zip_with

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

    Internal infrastructure can be setup when the iteration starts and can rely on the self.is_running to indicate when
    generation is expected. When this changes to False methods are expected to cleanup and exit gracefully, and be able
    to be called again with is_running set back to True. This allows restarting a complex stream object which may use
    threads requiring starting and stopping. The stop() method when called set is_running to False and attempts to call
    the same on self.src, this is meant to be used to stop any internal processes (ie. threads) when iteration stops
    with the expectation that it can be restarted later. Reading is_running or assigning a literal value to it is atomic
    thus thread-safe but keep this in mind when assigning a compound expression.
    """

    def __init__(self, src):
        """Initialize with `src' as the source iterable, and self.is_running as True."""
        self.src = src
        self.is_running = True

    def __iter__(self):
        """
        Iterate over every value from self.src, passing through self.generate() and yielding the
        values it generates.
        """
        self.is_running = True
        for src_val in self.src:
            for out_val in self.generate(src_val):
                yield out_val  # yield with syntax too new?

    def generate(self, val):
        """Generate values from input `val`, by default just yields that. """
        yield val

    def stop(self):
        """Sets self.is_running to False and calls stop() on self.src if it has this method."""
        self.is_running = False
        if callable(getattr(self.src, "stop", None)):
            self.src.stop()

    def get_gen_func(self):
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
        for out_val in self.func(val, *self.fargs, **self.fkwargs):
            yield out_val


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

    def __init__(self, src, buffer_size=None, order_type=OrderType.LINEAR):
        super().__init__(src)
        self.buffer_size = buffer_size
        self.order_type = order_type
        self.buffer = []

    def __iter__(self):
        self.buffer = [item for i, item in enumerate(self.src) if self.buffer_size is None or i < self.buffer_size]

        while self.is_running:
            inds = np.arange(0, len(self.buffer))

            if self.order_type == OrderType.SHUFFLE:
                np.random.shuffle(inds)
            elif self.order_type == OrderType.CHOICE:
                inds = np.random.choice(inds, len(self.buffer))

            for i in inds:
                for out_val in self.generate(self.buffer[i]):
                    yield out_val


@export
@alias("bufferstream")
class BufferStream(DataStream):
    """
    Accumulates a buffer of generated items, starting to yield them only when the buffer is filled and doing so until the
    buffer is empty. The buffer is filled by generate() which calls buffer_full() when full to allow subclasses to react.
    After this the buffer contents are yielded in order until the buffer is empty, then the filling process restarts.
    """

    def __init__(self, src, buffer_size=10, order_type=OrderType.LINEAR):
        super().__init__(src)
        self.buffer_size = buffer_size
        self.orderType = order_type
        self.buffer = []

    def buffer_full(self):
        """Called when the buffer is full and before emptying it."""

    def generate(self, val):
        if len(self.buffer) == self.buffer_size:
            self.buffer_full()  # call overridable callback to trigger action when buffer full

            if self.orderType == OrderType.SHUFFLE:
                np.random.shuffle(self.buffer)
            elif self.orderType == OrderType.CHOICE:
                inds = np.random.choice(np.arange(len(self.buffer)), len(self.buffer))
                self.buffer = [self.buffer[i] for i in inds]

            while len(self.buffer) > 0:
                yield self.buffer.pop(0)

        self.buffer.append(val)


@export
@alias("batchstream")
class BatchStream(BufferStream):
    """Collects values from the source together into a batch of the stated size, ie. stacks buffered items."""

    def __init__(self, src, batch_size, send_short_batch=False, order_type=OrderType.LINEAR):
        super().__init__(src, batch_size, order_type)
        self.send_short_batch = send_short_batch

    def buffer_full(self):
        """Replaces the buffer's contents with the arrays stacked together into a single item."""
        if isinstance(self.buffer[0], np.ndarray):
            # stack all the arrays together
            batch = np.stack(self.buffer)
        else:
            # stack the arrays from each item into one
            batch = tuple(zip_with(np.stack, *self.buffer))

        self.buffer[:] = [batch]  # yield only the one item when emptying the buffer

    def __iter__(self):
        for src_val in super().__iter__():
            yield src_val

        # only true if the iteration has completed but items are left to make up a shortened batch
        if len(self.buffer) > 0 and self.send_short_batch:
            self.buffer_full()
            yield self.buffer.pop()


@export
@alias("mergestream")
class MergeStream(DataStream):
    """Merge data from multiple iterators into generated tuples."""

    def __init__(self, *srcs):
        self.srcs = srcs
        super().__init__(RestartGenerator(self.yield_merged_values))

    def yield_merged_values(self):
        iters = [iter(s) for s in self.srcs]
        can_continue = True

        while self.is_running and can_continue:
            try:
                values = []
                for it in iters:
                    val = next(it)  # raises StopIteration when a source runs out of data at which point we quit

                    if not isinstance(val, (list, tuple)):
                        val = (val,)

                    values.append(tuple(val))

                src_val = sum(values, ())

                for out_val in self.generate(src_val):
                    yield out_val
            # must be caught as StopIteration won't propagate but magically mutate into RuntimeError
            except StopIteration:
                can_continue = False


@export
@alias("cyclingstream")
class CyclingStream(DataStream):

    def __init__(self, *srcs):
        self.srcs = srcs
        super().__init__(RestartGenerator(self.yield_alternating_values))

    def yield_alternating_values(self):
        iters = [iter(s) for s in self.srcs]
        can_continue = True

        while self.is_running and can_continue:
            try:
                for it in iters:
                    src_val = next(it)  # raises StopIteration when a source runs out of data at which point we quit
                    for out_val in self.generate(src_val):
                        yield out_val

            # must be caught as StopIteration won't propagate but magically mutate into RuntimeError
            except StopIteration:
                can_continue = False


@export
class PrefetchStream(DataStream):
    """
    Calculates item dtype and shape before iteration. This will get a value from `src` in the constructor, assign it to
    self.src_val, then assign the dtypes and shapes of the arrays to self.dtypes and self.shapes respectively. When it is
    iterated over self.src_val is yielded first followed by whatever else `src` produces so no data is lost.
    """

    def __init__(self, src):
        self.origSrc = src
        self.it = iter(src)
        self.src_val = next(self.it)

        if isinstance(self.src_val, np.ndarray):
            self.dtypes = self.src_val.dtype
            self.shapes = self.src_val.shape
        else:
            self.dtypes = tuple(b.dtype for b in self.src_val)
            self.shapes = tuple(b.shape for b in self.src_val)

        super().__init__(RestartGenerator(self._get_src))

    def _get_src(self):
        if self.it is not None:
            yield self.src_val
        else:
            self.it = iter(self.origSrc)  # self.it is None when restarting so recreate the iterator here

        for src_val in self.it:
            yield src_val

        self.it = None


@export
@alias("finitestream")
class FiniteStream(DataStream):
    """Yields only the specified number of items before quiting."""

    def __init__(self, src, num_items):
        super().__init__(src)
        self.num_items = num_items

    def __iter__(self):
        for _, item in zip(range(self.num_items), super().__iter__()):
            yield item


@export
@alias("tracestream")
class TraceStream(DataStream):

    def generate(self, val):
        vals = val if isinstance(val, (tuple, list)) else (val,)

        sizes = ", ".join("%s%s" % (s.dtype, s.shape) for s in vals)

        print("Stream -> %s" % sizes, flush=True)

        yield val
