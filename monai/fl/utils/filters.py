import abc

from monai.fl.utils.exchange_object import ExchangeObject


class Filter(abc.ABC):
    """
    Used to apply filter to content of ExchangeObject.
    """

    def __call__(self, data: ExchangeObject, extra=None) -> ExchangeObject:
        """
        Run the filtering.

        Arguments:
            data: ExchangeObject containing some data.

        Returns:
            ExchangeObject: filtered data.
        """

        raise NotImplementedError


class SummaryFilter(Filter):
    """
    Summary filter to content of ExchangeObject.
    """

    def __call__(self, data: ExchangeObject, extra=None) -> ExchangeObject:
        """
        Example filter that doesn't filter anything but only prints data summary.

        Arguments:
            data: ExchangeObject containing some data.

        Returns:
            ExchangeObject: filtered data.
        """

        print(f"Summary of ExchangeObject: {data.summary()}")

        return data
