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
        if not isinstance(data, dict):
            raise ValueError("'data' must be of type dict but is '{type(data)}'")

        rd = dict(data)
        for k in self.keys:
            rd[k] = apply_pending(rd[k], **kwargs)
        return rd

    def inverse(self, data):
        if not isinstance(data, dict):
            raise ValueError("'data' must be of type dict but is '{type(data)}'")

        return self(data)
