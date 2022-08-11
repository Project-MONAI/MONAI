from monai.fl.utils.exchange_object import ExchangeObject
from monai.fl.utils.filter import Filter


class DummyFilter(Filter):
    """
    Dummy filter to content of ExchangeObject.
    """

    def __call__(self, data: ExchangeObject, extra=None) -> ExchangeObject:
        """
        Dummy filter doesn't filter anything but only prints data summary.

        Arguments:
            data: ExchangeObject containing some data.

        Returns:
            ExchangeObject: filtered data.
        """

        print(f"Dummy filtering ExchangeObject: {data.summary()}")

        return data
