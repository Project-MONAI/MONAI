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

# Commonly used concepts are documented in this `type_definitions.py` file
# to explicitly identify what information that should be used consistently throughout
# the entire MONAI package. Type would be named  as type_definitions.DictKeySelection
# which states what we want to say in the name itself.
#
# The definitions in this file map context meaningful names to the underlying
# storage types.
#
# A conceptual type is represented by a new type name but is also one which
# can be different depending on environment. Consistent use of the concept
# and recorded documentation of the rationale and convention behind it lowers
# the learning curve for new developers. For readability, shorten names are
# preferred.
#

from typing import Hashable, Iterable, Union, Collection

# The KeysCollection type is used to for defining variables
# that store a subset of keys to select items from a dictionary.
# The container of keys must contain hashable elements.
KeysCollection = Collection[Hashable]

# The IndexSelection type is used to for defining variables
# that store a subset of indexes to select items from a List or Array like objects.
# The indexes must be integers, and if a container of indexes is specified, the
# container must be iterable.
IndexSelection = Union[Iterable[int], int]
