import monai
from monai.data.streams import OrderType
from .arrayreader import ArrayReader
import numpy as np


@monai.utils.export("monai.data.readers")
class NPZReader(ArrayReader):
    """
    Loads arrays from an .npz file as the source data. Other values can be loaded from the file and stored in 
    `other_values` rather than used as source data.
    """

    def __init__(self, obj_or_file_name, array_names, other_values=[],
                 order_type=OrderType.LINEAR, do_once=False, choice_probs=None):
        self.objOrFileName = obj_or_file_name

        dat = np.load(obj_or_file_name)

        keys = set(dat.keys())
        missing = set(array_names) - keys

        if missing:
            raise ValueError("Array name(s) %r not in loaded npz file" % (missing,))

        arrays = [dat[name] for name in array_names]

        super().__init__(*arrays, order_type=order_type, do_once=do_once, choice_probs=choice_probs)

        self.otherValues = {n: dat[n] for n in other_values if n in keys}
