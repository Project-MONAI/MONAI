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

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from parameterized import parameterized

from monai.utils import SKIP_EXTS, files_considered_equal, sources_equal

text1 = """
def foo(arg1, arg2):
    '''
    Compute the sum of arg1 and arg2 and return it.
    '''
    return arg1 + arg2
"""

text2 = """

def foo(arg1, arg2):
    '''
    Add arg1 and arg2 together and return it.
    '''
    return arg1 + arg2

"""

text3 = "def foo(arg1, arg2):return arg1 + arg2"

text4 = "def bar(arg1, arg2):return arg1 + arg2"

text5 = """
class Foo:
    def method(self,arg1):
        '''Returns arg1.'''
        return arg1
"""

TEXT_EQUAL_PAIRS = [(text1, text2), (text1, text3), (text2, text3), ("", "")]

TEXT_UNEQUAL_PAIRS = [(text1, text4), (text2, text4), (text3, text4), (text1, text5), (text1, "")]


class TestSourcesEqual(unittest.TestCase):
    @parameterized.expand(TEXT_EQUAL_PAIRS)
    def test_equal_texts(self, t1, t2):
        self.assertTrue(sources_equal(t1, t2))
        self.assertTrue(sources_equal(t2, t1))

    @parameterized.expand(TEXT_UNEQUAL_PAIRS)
    def test_unequal_texts(self, t1, t2):
        self.assertFalse(sources_equal(t1, t2))
        self.assertFalse(sources_equal(t2, t1))


class TestFilesEqual(unittest.TestCase):
    def test_equal_exts(self):
        """Test that two non-existent files with .py extensions result in an exception rather than False return."""
        with self.assertRaises(FileNotFoundError):
            files_considered_equal("/path/to/file1.py", "/path/to/somewhere/else/file2.py")

    def test_unequal_exts(self):
        """Test that two non-existent files with different extensions result in a False return."""
        self.assertFalse(files_considered_equal("/path/to/file1.text", "/path/to/file1.py"))

    @parameterized.expand(SKIP_EXTS)
    def test_skip_docs(self, ext):
        self.assertTrue(files_considered_equal("file1" + ext, "file2" + ext))

    @parameterized.expand(TEXT_EQUAL_PAIRS)
    def test_equal_text(self, t1, t2):
        with TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir, "t1.py")
            p2 = Path(tmpdir, "t2.py")
            p1.write_text(t1)
            p2.write_text(t2)

            self.assertTrue(files_considered_equal(p1, p2))

    @parameterized.expand(TEXT_UNEQUAL_PAIRS)
    def test_unequal_text(self, t1, t2):
        with TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir, "t1.py")
            p2 = Path(tmpdir, "t2.py")
            p1.write_text(t1)
            p2.write_text(t2)

            self.assertFalse(files_considered_equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
