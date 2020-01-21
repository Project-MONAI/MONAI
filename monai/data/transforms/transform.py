# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class Transform(object):
    """An abstract class of a ``Transform``
    A transform is callable that maps data into output data.
    """

    def __call__(self, data):
        """This method should return an updated version of ``data``.
        One useful case is to create multiple instances of this class and
        chain them together to form a more powerful transform:
            for transform in transforms:
                data = transform(data)
        Args:
            data (dict): an element which often comes from an iteration over an iterable,
                         such as``torch.utils.data.Dataset``
        """
        raise NotImplementedError
