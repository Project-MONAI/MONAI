from typing import Optional

class EnsembleMethods():
    def __init__(self,
        validation_data, 
        test_data, 
        model_paths
    ):
        self.validation_data = validation_data
        self.test_data = test_data

    def predict(self, print_results: bool=True, save_predictions: bool=True):
        """
        predict results after the models are ranked/weighted
        """
        raise NotImplementedError

    def get_model(self, model_path, weights: Optional[float] = None):
        """
        register model in the ensemble
        """
        pass


    def get_validation_result(self, performances):
        raise NotImplementedError


    def rank_model(self):
        raise NotImplementedError


class EnsembleBestN():
    def __init__(self, *args, n_best=5):
        super().__init__(*args)    

    def get_validation_result(self):
        pass

    def rank_model(self):
        pass

    def predict(self):
        pass

class EnsembleWeightedBestN():
    def __init__(self, *args, n_best=5):
        super().__init__(*args)    

    def get_validation_result(self):
        pass

    def rank_model(self):
        pass

    def predict(self):
        pass