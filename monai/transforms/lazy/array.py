from monai.data.meta_tensor import MetaTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.lazy.functional import apply_pending


class ApplyPending(InvertibleTransform):
    """
    Apply wraps the apply_pending method and can function as a Transform in an array-based pipeline.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data, *args, **kwargs):
        return apply_pending(data, *args, **kwargs)

    def inverse(self, data):
        return self(data)
