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

from fed_learn.server.fed_server import FederatedServer
from fed_learn.components.pt_model_saver import PTModelSaver
from fed_learn.server.model_aggregator import ModelAggregator


class ServerTrainer:

    def __init__(
        self,
        model_options,
    ):
        self.ckpt_preload_path = model_options['MMAR_CKPT'] if 'MMAR_CKPT' in model_options else None
        self.secure_train = model_options['secure_train']
        self.server_config = []
        self.model_log_dir = None
        self.model_saver_options = None

        self.model_log_dir = model_options['MMAR_CKPT_DIR']

    def set_server_config(self, server_config):
        self.server_config = server_config

    def set_model_saver_options(self, options):
        self.model_saver_options = options

    def train(self):
        services = self.deploy()
        self.start_training(services)

    def start_training(self, services):
        services.start()

    def deploy(self):

        # We only deploy the first server right now
        first_server = sorted(self.server_config)[0]

        # Dynamically create model saver.
        model_saver = PTModelSaver(
            exclude_vars=first_server['exclude_vars'],
            model_log_dir=self.model_log_dir,
            ckpt_preload_path=self.ckpt_preload_path
        )

        aggregator = ModelAggregator(
            exclude_vars=first_server['exclude_vars'],
            aggregation_weights={"1": 1.0, "2": 1.0}
        )

        # only use the first server
        services = FederatedServer(
            task_name=first_server['name'],
            min_num_clients=first_server['min_num_clients'],
            max_num_clients=first_server['max_num_clients'],
            start_round=first_server['start_round'],
            num_rounds=first_server['num_rounds'],
            exclude_vars=first_server['exclude_vars'],
            model_log_dir=self.model_log_dir,
            model_saver=model_saver,
            model_aggregator=aggregator
        )

        services.deploy(grpc_args=first_server, secure_train=self.secure_train)
        return services

    def close(self):
        pass
