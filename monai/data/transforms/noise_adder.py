from .multi_format_transformer import MultiFormatTransformer


class NoiseAdder(MultiFormatTransformer):
    """Adds noise to the entire image.

    Args:
        No argument
    """

    def __init__(self, noise):
        MultiFormatTransformer.__init__(self)
        self.noise = noise

    def _handle_any(self, img):
        return img + self.noise
