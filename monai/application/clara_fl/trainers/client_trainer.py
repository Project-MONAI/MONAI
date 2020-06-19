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
        fitter = SupervisedFitter(self.num_epochs)

        servers = [{t['name']: t['service']} for t in self.server_config]
        federated_client = FederatedClient(
            client_id=str(self.uid),
            # We only deploy the first server right now .....
            server_args=sorted(servers)[0],
            client_args=self.client_config,
            exclude_vars=self.client_config['exclude_vars'],
            secure_train=self.secure_train,
            model_reader_writer=PTModelReaderWriter()
        )
        return federated_client.run(fitter)

    def close(self):
        pass
