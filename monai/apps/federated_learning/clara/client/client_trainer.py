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

from fed_learn.client.fed_client import FederatedClient
from fed_learn.components.pt_model_reader_writer import PTModelReaderWriter
from .supervised_fitter import SupervisedFitter


class ClientTrainer:

    def __init__(self,
                 uid,
                 num_epochs,
                 server_config,
                 client_config,
                 secure_train):
        self.server_config = server_config
        self.client_config = client_config
        self.uid = uid
        self.num_epochs = num_epochs
        self.secure_train = secure_train

    def train(self):
        fitter = self.deploy()
        federated_client = self.create_fed_client()
        federated_client.start_heartbeat()
        return federated_client.run(fitter)

    def deploy(self):
        return SupervisedFitter(self.num_epochs)

    def create_fed_client(self):
        servers = [{t['name']: t['service']} for t in self.server_config]
        return FederatedClient(
            client_id=str(self.uid),
            # We only deploy the first server right now .....
            server_args=sorted(servers)[0],
            client_args=self.client_config,
            exclude_vars=self.client_config['exclude_vars'],
            secure_train=self.secure_train,
            model_reader_writer=PTModelReaderWriter()
        )

    def close(self):
        pass
