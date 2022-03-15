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

__all__ = ["ID_REF_KEY", "ID_SEP_KEY", "EXPR_KEY", "MACRO_KEY"]


ID_REF_KEY = "@"  # start of a reference to a ConfigItem
ID_SEP_KEY = "#"  # separator for the ID of a ConfigItem
EXPR_KEY = "$"  # start of a ConfigExpression
MACRO_KEY = "%"  # start of a macro of a config
