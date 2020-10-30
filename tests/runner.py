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

import time
import unittest


class TimeLoggingTestRunner(unittest.runner.TextTestRunner):
    """Overload the default runner so that we can get and print the results."""

    def __init__(self, *args, **kwargs):
        return super().__init__(resultclass=TimeLoggingTestResult, *args, **kwargs)

    def run(self, test):
        result = super().run(test)
        print("\n\ntests finished, printing times in descending order...\n")
        timings = dict(sorted(result.getTestTimings().items(), key=lambda kv: kv[1], reverse=True))
        for r in timings:
            print(f"{r} ({timings[r]:.03}s)")
        return result


class TimeLoggingTestResult(unittest.TextTestResult):
    """Overload the default results so that we can store the results."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timed_tests = dict()

    def startTest(self, test):  # noqa: N802
        """Start timer, print test name, do normal test."""
        self._started_at = time.time()
        name = self.getDescription(test)
        self.stream.write(f"{name}...")
        super().startTest(test)

    def stopTest(self, test):  # noqa: N802
        """On test end, get time, print, store and do normal behaviour."""
        elapsed = time.time() - self._started_at
        self.stream.write(f"({elapsed:.03}s)\n")
        name = self.getDescription(test)
        if name in self.timed_tests:
            raise AssertionError("expected all keys to be unique")
        self.timed_tests[name] = elapsed
        super().stopTest(test)

    def getTestTimings(self):  # noqa: N802
        """Get all times so they can be sorted and printed."""
        return self.timed_tests


if __name__ == "__main__":
    loader = unittest.TestLoader()
    tests = loader.discover(".")
    test_runner = TimeLoggingTestRunner()
    test_result = test_runner.run(tests)
