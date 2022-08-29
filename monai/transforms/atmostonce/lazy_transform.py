from monai.config import NdarrayOrTensor
from monai.data import MetaTensor
from monai.transforms import Randomizable
from monai.transforms.atmostonce.apply import Applyd
from monai.utils.mapping_stack import MetaMatrix


# TODO: move to mapping_stack.py
def push_transform(
        data: MetaTensor,
        meta_matrix: MetaMatrix
):
    data.push_pending_transform(meta_matrix)


# TODO: move to mapping_stack.py
def update_metadata(
        data: MetaTensor,
        transform: NdarrayOrTensor,
        extra_info
):
    pass


# TODO: move to utils
def flatten_sequences(seq):

    def flatten_impl(s, accum):
        if isinstance(s, (list, tuple)):
            for inner_t in s:
                accum = flatten_impl(inner_t, accum)
        else:
            accum.append(s)
        return accum

    dest = []
    for s in seq:
        dest = flatten_impl(s, dest)

    return dest


def transforms_compatible(current, next):
    raise NotImplementedError()


def compile_lazy_transforms(transforms):
    flat = flatten_sequences(transforms)
    for i in range(len(flat)-1):
        cur_t, next_t = flat[i], flat[i + 1]
        if not transforms_compatible(cur_t, next_t):
            flat.insert(i + 1, Applyd())
    if not isinstance(flat[-1], Applyd):
        flat.append(Applyd)
    return flat

def compile_cached_dataloading_transforms(transforms):
    flat = flatten_sequences(transforms)
    for i in range(len(flat)):
        cur_t = flat[i]
        if isinstance(cur_t, Randomizable):
            flat.insert




class LazyTransform:

    def __init__(self, lazy_evaluation):
        self.lazy_evaluation = lazy_evaluation

    # TODO: determine whether to have an 'eval' defined here that implements laziness
    # def __call__(self, *args, **kwargs):
    #     """Call this method after calculating your meta data"""
    #     if self.lazily_evaluate:
    #         # forward the transform to metatensor
    #         pass
    #     else:
    #         # apply the transform and reset the stack on metatensor
    #         pass
