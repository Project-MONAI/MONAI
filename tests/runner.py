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

import argparse
import os
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


def print_results(results, discovery_time, thresh, status):
    # only keep results >= threshold
    results = dict(filter(lambda x: x[1] > thresh, results.items()))
    if len(results) == 0:
        return
    print(f"\n\n{status}, printing completed times >{thresh}s in ascending order...\n")
    timings = dict(sorted(results.items(), key=lambda item: item[1]))

    for r in timings:
        if timings[r] >= thresh:
            print(f"{r} ({timings[r]:.03}s)")
    print(f"test discovery time: {discovery_time:.03}s")
    print(f"total testing time: {sum(results.values()):.03}s")
    print("Remember to check above times for any errors!")


def parse_args():
    parser = argparse.ArgumentParser(description="Runner for MONAI unittests with timing.")
    parser.add_argument(
        "-s", action="store", dest="path", default=".", help="Directory to start discovery ('.' default)"
    )
    parser.add_argument(
        "-p", action="store", dest="pattern", default=None, help="Pattern to match tests (default is unittest default)"
    )
    parser.add_argument(
        "-t",
        "--thresh",
        dest="thresh",
        default=10.0,
        type=float,
        help="Display tests longer than given threshold default: 10)",
    )
    parser.add_argument("-q", "--quick", action="store_true", dest="quick", default=False, help="Only do quick tests")
    args = parser.parse_args()
    print(f"Running tests in folder: '{args.path}'")
    if args.pattern:
        print(f"With file pattern: '{args.pattern}'")

    return args.path, args.pattern, args.thresh, args.quick


if __name__ == "__main__":
    # Parse input arguments
    path, pattern, thresh, quick = parse_args()

    # If quick is desired, set environment variable
    if any(q in sys.argv for q in ["-q", "--quick"]):
        os.environ["QUICKTEST"] = "True"

    # Get all test names (optionally from some path with some pattern)
    loader_args = [path, pattern] if pattern else [path]
    loader = unittest.TestLoader()
    with PerfContext() as pc:
        tests = loader.discover(*loader_args)
    discovery_time = pc.total_time
    print(f"time to discover tests: {discovery_time}s")

    test_runner = unittest.runner.TextTestRunner(resultclass=TimeLoggingTestResult)

    # Use try catches to print the current results if encountering exception or keyboard interruption
    try:
        test_result = test_runner.run(tests)
        print_results(results, discovery_time, thresh, "tests finished")
        sys.exit(not test_result.wasSuccessful())
    except KeyboardInterrupt:
        print_results(results, discovery_time, thresh, "tests cancelled")
        sys.exit(1)
    except Exception:
        print_results(results, discovery_time, thresh, "exception reached")
        raise
