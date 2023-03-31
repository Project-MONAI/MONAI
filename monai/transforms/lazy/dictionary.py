from monai.transforms.inverse import InvertibleTransform


class ApplyPendingd(InvertibleTransform):
    """
    Apply wraps the apply method and can function as a Transform in either array or dictionary
    mode.
    """

    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __call__(self, data, **kwargs):
        rd = dict(data)
        for k in self.keys:
            rd[k] = apply_pending(rd[k], **kwargs)
        return rd

        # return apply_transforms(data, *args, **kwargs)

    def inverse(self, data):
        return self(data)
    