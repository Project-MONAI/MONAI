from monai.transforms.inverse import InvertibleTransform
from monai.transforms.lazy.functional import apply_pending

__all__ = ["ApplyPendingd"]


class ApplyPendingd(InvertibleTransform):
    """
    Apply wraps the apply method and can function as a Transform in either array or dictionary
    mode.
    """

    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __call__(self, data, *args, **kwargs):
        rd = dict(data)
        for k in self.keys:
            rd[k] = apply_pending(rd[k], *args, **kwargs)
        return rd

        # return apply_transforms(data, *args, **kwargs)

    def inverse(self, data):
        return self(data)
