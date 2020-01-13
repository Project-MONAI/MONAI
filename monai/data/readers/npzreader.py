
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
from monai.data.streams import OrderType
from .arrayreader import ArrayReader
import numpy as np


@monai.utils.export("monai.data.readers")
class NPZReader(ArrayReader):
    """
    Loads arrays from an .npz file as the source data. Other values can be loaded from the file and stored in 
    `otherValues` rather than used as source data.
    """

    def __init__(self, objOrFileName, arrayNames, otherValues=[], 
                 orderType=OrderType.LINEAR, doOnce=False, choiceProbs=None):
        self.objOrFileName = objOrFileName

        dat = np.load(objOrFileName)

        keys = set(dat.keys())
        missing = set(arrayNames) - keys

        if missing:
            raise ValueError("Array name(s) %r not in loaded npz file" % (missing,))

        arrays = [dat[name] for name in arrayNames]

        super().__init__(*arrays, orderType=orderType, doOnce=doOnce, choiceProbs=choiceProbs)

        self.otherValues = {n: dat[n] for n in otherValues if n in keys}
