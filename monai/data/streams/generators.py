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

from glob import glob

import numpy as np

from monai.data.readers.arrayreader import ArrayReader
from monai.data.streams.datastream import OrderType
from monai.utils.moduleutils import export


@export("monai.data.streams")
class GlobPathGenerator(ArrayReader):
    """
    Generates file paths from given glob patterns, expanded using glob.glob. This will yield the file names as tuples
    of strings, if multiple patterns are given the a file from each expansion is yielded in the tuple.
    """

    def __init__(self, *glob_paths, sort_paths=True, order_type=OrderType.LINEAR, do_once=False, choice_probs=None):
        """
        Construct the generator using the given glob patterns `glob_paths`. If `sort_paths` is True each list of files
        is sorted independently. 

        Args:
            glob_paths (list of str): list of glob patterns to expand
            sort_paths (bool): if True, each file list is sorted
            order_type (OrderType): the type of order to yield tuples in
            do_once (bool): if True, the list of files is iterated through only once, indefinitely loops otherwise
            choice_probs (np.ndarray): list of per-item probabilities for OrderType.CHOICE
        """

        expanded_paths = list(map(glob, glob_paths))
        if sort_paths:
            expanded_paths = list(map(sorted, expanded_paths))

        expanded_paths = list(map(np.asarray, expanded_paths))

        super().__init__(*expanded_paths, order_type=order_type, do_once=do_once, choice_probs=choice_probs)
        self.glob_paths = glob_paths
        self.sort_paths = sort_paths
