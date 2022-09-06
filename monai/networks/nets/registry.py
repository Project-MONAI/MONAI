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

from monai.networks.nets import BasicEncoder

__all__ = ["Register"]


class Register:
    """
    A register to regist backbones for the flexible unet.
    """

    def __init__(self):
        self.register_dict = {}

    def regist_class(self, name):
        if not isinstance(name, BasicEncoder):
            raise Exception("An encoder must derive from BasicEncoder class.")
        else:
            name_string_list = name._get_encoder_name_string_list()
            feature_number_list = name._get_output_feature_number_list()
            feature_channel_list = name._get_output_feature_channel_list()
            parameter_list = name._get_parameter()

            assert len(name_string_list) == len(feature_number_list) == len(feature_channel_list) == len(parameter_list)
            for cnt, name_string in enumerate(name_string_list):
                cur_dict = {
                    "type": name,
                    "feature_number": feature_number_list[cnt],
                    "feature_channel": feature_channel_list[cnt],
                    "parameter": parameter_list[cnt],
                }
                self.register_dict[name_string] = cur_dict


BACKBONE = Register()
