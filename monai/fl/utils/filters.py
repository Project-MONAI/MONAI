# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc

from monai.fl.utils.exchange_object import ExchangeObject


class Filter(abc.ABC):
    """
    Used to apply filter to content of ExchangeObject.
    """

    @abc.abstractmethod
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
