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


class Configurator:
    def __init__(
        self,
        data_stats_filename: str = None,
        input_filename: str = None,
    ):
        self.data_stats_filename = data_stats_filename
        self.input_filename = input_filename

    def load(self):
        pass

    def update(self):
        pass

    def write(self):
        pass
