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

import argparse
import glob
import inspect
import os
import re
import sys
import time
import unittest

from monai.utils import PerfContext

results: dict = {}


class TimeLoggingTestResult(unittest.TextTestResult):
    """Overload the default results so that we can store the results."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timed_tests = {}

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
        "-s", action="store", dest="path", default=".", help="Directory to start discovery (default: '%(default)s')"
    )
    parser.add_argument(
        "-p",
        action="store",
        dest="pattern",
        default="test_*.py",
        help="Pattern to match tests (default: '%(default)s')",
    )
    parser.add_argument(
        "-t",
        "--thresh",
        dest="thresh",
        default=10.0,
        type=float,
        help="Display tests longer than given threshold (default: %(default)d)",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="store",
        dest="verbosity",
        type=int,
        default=1,
        help="Verbosity level (default: %(default)d)",
    )
    parser.add_argument("-q", "--quick", action="store_true", dest="quick", default=False, help="Only do quick tests")
    parser.add_argument(
        "-f", "--failfast", action="store_true", dest="failfast", default=False, help="Stop testing on first failure"
    )
    args = parser.parse_args()
    print(f"Running tests in folder: '{args.path}'")
    if args.pattern:
        print(f"With file pattern: '{args.pattern}'")

    return args


def get_default_pattern(loader):
    signature = inspect.signature(loader.discover)
    params = {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}
    return params["pattern"]


if __name__ == "__main__":

    # Parse input arguments
    args = parse_args()

    # If quick is desired, set environment variable
    if args.quick:
        os.environ["QUICKTEST"] = "True"

    # Get all test names (optionally from some path with some pattern)
    with PerfContext() as pc:
        # the files are searched from `tests/` folder, starting with `test_`
        files = glob.glob(os.path.join(os.path.dirname(__file__), "test_*.py"))
        cases = []
        for test_module in {os.path.basename(f)[:-3] for f in files}:
            if re.match(args.pattern, test_module):
                cases.append(f"tests.{test_module}")
            else:
                print(f"monai test runner: excluding tests.{test_module}")
        tests = unittest.TestLoader().loadTestsFromNames(cases)
    discovery_time = pc.total_time
    print(f"time to discover tests: {discovery_time}s, total cases: {tests.countTestCases()}.")

    test_runner = unittest.runner.TextTestRunner(
        resultclass=TimeLoggingTestResult, verbosity=args.verbosity, failfast=args.failfast
    )

    # Use try catches to print the current results if encountering exception or keyboard interruption
    try:
        test_result = test_runner.run(tests)
        print_results(results, discovery_time, args.thresh, "tests finished")
        sys.exit(not test_result.wasSuccessful())
    except KeyboardInterrupt:
        print_results(results, discovery_time, args.thresh, "tests cancelled")
        sys.exit(1)
    except Exception:
        print_results(results, discovery_time, args.thresh, "exception reached")
        raise
