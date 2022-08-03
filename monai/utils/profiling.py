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

import csv
import datetime
import multiprocessing
import os
import sys
import threading
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import wraps
from inspect import getframeinfo, stack
from queue import Empty
from time import perf_counter, perf_counter_ns

import numpy as np
import torch

from monai.utils import optional_import

pd, has_pandas = optional_import("pandas")

__all__ = [
    "torch_profiler_full",
    "torch_profiler_time_cpu_gpu",
    "torch_profiler_time_end_to_end",
    "PerfContext",
    "WorkflowProfiler",
    "ProfileHandler",
    "select_transform_call",
]


def torch_profiler_full(func):
    """
    A decorator which will run the torch profiler for the decorated function,
    printing the results in full.
    Note: Enforces a gpu sync point which could slow down pipelines.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            result = func(*args, **kwargs)

        print(prof, flush=True)

        return result

    return wrapper


def torch_profiler_time_cpu_gpu(func):
    """
    A decorator which measures the execution time of both the CPU and GPU components
    of the decorated function, printing both results.
    Note: Enforces a gpu sync point which could slow down pipelines.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            result = func(*args, **kwargs)

        cpu_time = prof.self_cpu_time_total
        gpu_time = sum(evt.self_cuda_time_total for evt in prof.function_events)

        cpu_time = torch.autograd.profiler.format_time(cpu_time)
        gpu_time = torch.autograd.profiler.format_time(gpu_time)

        print(f"cpu time: {cpu_time}, gpu time: {gpu_time}", flush=True)

        return result

    return wrapper


def torch_profiler_time_end_to_end(func):
    """
    A decorator which measures the total execution time from when the decorated
    function is called to when the last cuda operation finishes, printing the result.
    Note: Enforces a gpu sync point which could slow down pipelines.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        torch.cuda.synchronize()
        start = perf_counter()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        end = perf_counter()

        total_time = (end - start) * 1e6
        total_time_str = torch.autograd.profiler.format_time(total_time)
        print(f"End-to-end time: {total_time_str}", flush=True)

        return result

    return wrapper


class PerfContext:
    """
    Context manager for tracking how much time is spent within context blocks. This uses `time.perf_counter` to
    accumulate the total amount of time in seconds in the attribute `total_time` over however many context blocks
    the object is used in.
    """

    def __init__(self):
        self.total_time = 0
        self.start_time = None

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.total_time += perf_counter() - self.start_time
        self.start_time = None


# stores the results from profiling with trace or with other helper methods
ProfileResult = namedtuple("ProfileResult", ["name", "time", "filename", "lineno", "pid", "timestamp"])


def select_transform_call(frame):
    """Returns True if `frame` is a call to a `Transform` object's `_call__` method."""
    from monai.transforms import Transform  # prevents circular import

    self_obj = frame.f_locals.get("self", None)
    return frame.f_code.co_name == "__call__" and isinstance(self_obj, Transform)


class WorkflowProfiler:
    """
    Profiler for timing all aspects of a workflow. This includes using stack tracing to capture call times for
    all selected calls (by default calls to `Transform.__call__` methods), times within context blocks, times
    to generate items from iterables, and times to execute decorated functions.

    This profiler must be used only within its context because it uses an internal thread to read results from a
    multiprocessing queue, this allows the profiler to function across multiple threads and processes. The
    profiler uses `sys.settrace` and `threading.settrace` to find all calls to profile, this will be set when
    the context enters and cleared when it exits so proper use of the context is essential to prevent excessive
    tracing. Note that tracing has a high overhead so times will not accurately reflect real world performance
    but give an idea of relative share of time spent.

    The tracing functionality uses a selector to choose which calls to trace, since tracing all calls induces
    infinite loops and would be terribly slow even if not. This selector is a callable accepting a `call` trace
    frame and returns True if the call should be traced. The dedault is `select_transform_call` which will return
    True for `Transform.__call__` calls only.

    Example showing use of all profiling functions:

    .. code-block:: python

        import monai.transform as mt
        comp=mt.Compose([mt.ScaleIntensity(),mt.RandAxisFlip(0.5)])

        with WorkflowProfiler() as wp:
            for _ in wp.profile_iter("range",range(5)):
                with wp.profile_ctx("Loop"):
                    for i in range(10):
                        comp(torch.rand(1,16,16))

            @wp.profile_callable()
            def foo(): pass

            foo()
            foo()

        print(wp.get_times_summary_pd())  # print results

    Args:
        call_selector: selector to determine which calls to trace, use None to disable tracing
    """

    def __init__(self, call_selector=select_transform_call):
        self.results = defaultdict(list)
        self.parent_pid = os.getpid()
        self.read_thread = None
        self.lock = threading.RLock()
        self.queue = multiprocessing.SimpleQueue()
        self.queue_timeout = 0.1
        self.call_selector = call_selector

    def _is_parent(self):
        """Return True if this is the parent process."""
        return os.getpid() == self.parent_pid

    def _is_thread_active(self):
        """Return True if the read thread should be still active."""
        return self.read_thread is not None or not self.queue.empty()

    def _read_thread_func(self):
        """Read results from the queue and add to self.results in a thread stared by `__enter__`."""
        while self._is_parent() and self._is_thread_active():
            try:
                result = self.queue.get()

                if result is None:
                    break

                self.add_result(result)
            except Empty:
                pass

        assert not self._is_parent() or self.queue.empty()

    def _put_result(self, name, timedelta, filename, lineno):
        """Add a ProfileResult object to the queue."""
        ts = str(datetime.datetime.now())
        self.queue.put(ProfileResult(name, timedelta, filename, lineno, os.getpid(), ts))

    def _trace_call(self, frame, why, arg):
        """
        Trace calls, when a call is encountered that is accepted by self.call_selector, create a new function to
        trace that call and measure the time from the call to a "return" frame.
        """
        if why == "call":
            if self.call_selector(frame):
                calling_frame = frame
                start = perf_counter_ns()

                def _call_profiler(frame, why, arg):
                    """Defines a new inner trace function just for this call."""
                    if why == "return":
                        diff = perf_counter_ns() - start
                        f_code = calling_frame.f_code
                        self_obj = calling_frame.f_locals.get("self", None)
                        name = f_code.co_name
                        if self_obj is not None:
                            name = f"{type(self_obj).__name__}.{name}"

                        self._put_result(name, diff, f_code.co_filename, f_code.co_firstlineno)

                # This function will be used to trace this specific call now, however any new functions calls
                # within will cause a "call" frame to be sent to `_trace_call` rather than to it, ie. it's not
                # actually recursively tracing everything below as the documentation suggests and so cannot
                # control whether subsequence calls are traced (see https://bugs.python.org/issue11992).
                return _call_profiler
        else:
            return self._trace_call

    def __enter__(self):
        """Enter the context, creating the read thread and setting up tracing if needed."""
        self.read_thread = threading.Thread(target=self._read_thread_func)
        self.read_thread.start()

        if self.call_selector is not None:
            threading.settrace(self._trace_call)
            sys.settrace(self._trace_call)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Terminate the read thread cleanly and reset tracing if needed."""
        assert self._is_parent()

        self.queue.put(None)

        read_thread = self.read_thread
        self.read_thread = None

        read_thread.join()

        if self.call_selector is not None:
            threading.settrace(None)
            sys.settrace(None)

    def add_result(self, result: ProfileResult):
        """Add a result in a thread-safe manner to the internal results dictionary."""
        with self.lock:
            self.results[result.name].append(result)

    def get_results(self):
        """Get a fresh results dictionary containing fresh tuples of ProfileResult objects."""
        if not self._is_parent():
            raise RuntimeError("Only parent process can collect results")

        with self.lock:
            return {k: tuple(v) for k, v in self.results.items()}

    @contextmanager
    def profile_ctx(self, name, caller=None):
        """Creates a context to profile, placing a timing result onto the queue when it exits."""
        if caller is None:
            caller = getframeinfo(stack()[2][0])  # caller of context, not something in contextlib

        start = perf_counter_ns()
        try:
            yield
        finally:
            diff = perf_counter_ns() - start
            self._put_result(name, diff, caller.filename, caller.lineno)

    def profile_callable(self, name=None):
        """
        Decorator which can be applied to a function which profiles any calls to it. All calls to decorated
        callables must be done within the context of the profiler.
        """

        def _outer(func):
            _name = func.__name__ if name is None else name
            return self.profile_ctx(_name)(func)

        return _outer

    def profile_iter(self, name, iterable):
        """Wrapper around anything iterable to profile how long it takes to generate items."""

        class _Iterable:
            def __iter__(_self):  # noqa: B902, N805 pylint: disable=E0213
                do_iter = True
                orig_iter = iter(iterable)
                caller = getframeinfo(stack()[1][0])

                while do_iter:
                    try:
                        start = perf_counter_ns()
                        item = next(orig_iter)
                        diff = perf_counter_ns() - start
                        # don't put result when StopIteration is hit
                        self._put_result(name, diff, caller.filename, caller.lineno)
                        yield item
                    except StopIteration:
                        do_iter = False

        return _Iterable()

    def get_times_summary(self, times_in_s=True):
        """
        Returns a dictionary mapping results entries to tuples containing the number of items, time sum, time average,
        time std dev, time min, and time max.
        """
        result = {}
        for k, v in self.get_results().items():
            timemult = 1e-9 if times_in_s else 1.0
            all_times = [res.time * timemult for res in v]

            timesum = sum(all_times)
            timeavg = timesum / len(all_times)
            timestd = np.std(all_times)
            timemin = min(all_times)
            timemax = max(all_times)

            result[k] = (len(v), timesum, timeavg, timestd, timemin, timemax)

        return result

    def get_times_summary_pd(self, times_in_s=True):
        """Returns the same informatoin as `get_times_summary` but in a Pandas DataFrame."""
        import pandas as pd

        summ = self.get_times_summary(times_in_s)
        suffix = "s" if times_in_s else "ns"
        columns = ["Count", f"Total Time ({suffix})", "Avg", "Std", "Min", "Max"]

        df = pd.DataFrame.from_dict(summ, orient="index", columns=columns)
        df = df.sort_values(columns[1], ascending=False)
        return df

    def dump_csv(self, stream=sys.stdout):
        """Save all results to a csv file."""
        all_results = list(self.get_results().values())
        writer = csv.DictWriter(stream, fieldnames=all_results[0][0]._asdict().keys())
        writer.writeheader()

        for rlist in all_results:
            for r in rlist:
                writer.writerow(r._asdict())


class ProfileHandler:
    """
    Handler for Ignite Engine classes which measures the time from a start event ton an end event. This can be used to
    profile epoch, iteration, and other events as defined in `ignite.engine.Events`. This class should be used only
    within the context of a profiler object.

    Args:
        name: name of event to profile
        profiler: instance of WorkflowProfiler used by the handler, should be within the context of this object
        start_event: item in `ignite.engine.Events` stating event at which to start timing
        end_event: item in `ignite.engine.Events` stating event at which to stop timing
    """

    def __init__(self, name: str, profiler: WorkflowProfiler, start_event, end_event):
        self.name = name
        self.profiler = profiler
        self.start_event = start_event
        self.end_event = end_event
        self.ctx = None

    def attach(self, engine):
        engine.add_event_handler(self.start_event, self.start)
        engine.add_event_handler(self.end_event, self.end)
        return self

    def start(self, engine):
        self.ctx = self.profiler.profile_ctx(self.name)
        self.ctx.__enter__()

    def end(self, engine):
        self.ctx.__exit__(None, None, None)
        self.ctx = None
