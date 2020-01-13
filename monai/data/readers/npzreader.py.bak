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
