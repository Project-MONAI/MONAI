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

import time
from functools import wraps

import torch

__all__ = ["torch_profiler_full", "torch_profiler_time_cpu_gpu", "torch_profiler_time_end_to_end", "PerfContext"]


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
        start = time.perf_counter()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        end = time.perf_counter()

        total_time = (end - start) * 1e6
        total_time_str = torch.autograd.profiler.format_time(total_time)
        print(f"end to end time: {total_time_str}", flush=True)

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
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.total_time += time.perf_counter() - self.start_time
        self.start_time = None
