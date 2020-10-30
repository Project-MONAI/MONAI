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

import sys
import time
import unittest

from monai.utils import PerfContext

results: dict = dict()


class TimeLoggingTestResult(unittest.TextTestResult):
    """Overload the default results so that we can store the results."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timed_tests = dict()

    def startTest(self, test):  # noqa: N802
        """Start timer, print test name, do normal test."""
        self.start_time = time.time()
        name = self.getDescription(test)
        self.stream.write(f"Starting test: {name}...\n")
        super().startTest(test)

    def stopTest(self, test):  # noqa: N802
        """On test end, get time, print, store and do normal behaviour."""
        elapsed = time.time() - self.start_time
        name = self.getDescription(test)
        self.stream.write(f"Finished test: {name} ({elapsed:.03}s)\n")
        if name in results:
            raise AssertionError("expected all keys to be unique")
        results[name] = elapsed
        super().stopTest(test)


def print_results(results, discovery_time):
    timings = dict(sorted(results.items(), key=lambda item: item[1]))
    for r in timings:
        print(f"{r} ({timings[r]:.03}s)")
    print(f"test discovery time: {discovery_time:.03}s")
    print(f"total testing time: {sum(results.values()):.03}s")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tests"
    print(f"Running tests in folder: '{path}'")

    loader = unittest.TestLoader()

    with PerfContext() as pc:
        tests = loader.discover(path)
    discovery_time = pc.total_time
    print(f"time to discover tests: {discovery_time}s")

    test_runner = unittest.runner.TextTestRunner(resultclass=TimeLoggingTestResult)

    try:
        test_result = test_runner.run(tests)
        print("\n\ntests finished, printing times in ascending order...\n")
        print_results(results, discovery_time)
    except KeyboardInterrupt:
        print("\n\ntests cancelled, printing completed times in ascending order...\n")
        print_results(results, discovery_time)
        exit(1)
    except Exception:
        print("\n\nexception reached, printing completed times in ascending order...\n")
        print_results(results, discovery_time)
        raise
