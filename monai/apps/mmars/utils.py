# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib


def get_class(class_path: str):
    """
    Get the class from specified class path.

    Args:
        class_path (str): full path of the class.

    Raises:
        ValueError: invalid class_path, missing the module name.
        ValueError: class does not exist.
        ValueError: module does not exist.

    """
    if len(class_path.split(".")) < 2:
        raise ValueError(f"invalid class_path: {class_path}, missing the module name.")
    module_name, class_name = class_path.rsplit(".", 1)

    try:
        module_ = importlib.import_module(module_name)

        try:
            class_ = getattr(module_, class_name)
        except AttributeError as e:
            raise ValueError(f"class {class_name} does not exist.") from e

    except AttributeError as e:
        raise ValueError(f"module {module_name} does not exist.") from e

    return class_


def instantiate_class(class_path: str, **kwargs):
    """
    Method for creating an instance for the specified class.

    Args:
        class_path: full path of the class.
        kwargs: arguments to initialize the class instance.

    Raises:
        ValueError: class has paramenters error.
    """

    try:
        return get_class(class_path)(**kwargs)
    except TypeError as e:
        raise ValueError(f"class {class_path} has parameters error.") from e
