class Ensemble():
    def __init__(self, validation_data, test_data, model_paths):
        self.validation_data = validation_data
        self.test_data = test_data

    def predict(self):
        raise NotImplementedError

    def get_model(self, model_path):
        pass

    def get_model_with_weights(self, model_path):
        pass

    def get_validation_result(self):
        raise NotImplementedError


class EnsembleBestN():
    def __init__(self, *args, n_best=5):
        super().__init__(*args)

    def predict(self):
        pass

    def get_validation_result(self):
        pass
