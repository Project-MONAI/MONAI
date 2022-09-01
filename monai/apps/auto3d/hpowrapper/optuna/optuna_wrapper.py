import optuna
import sys
sys.path.append('..')
from dummy_trainner import trainner
from hpo_wrapper import HPO_wrapper

class OPTUNA_wrapper(HPO_wrapper):
    def __init__(self, trainner=None, **kwargs):
        super().__init__(trainner, **kwargs)

    def _get_hyperparameters(self, trial):
        return {'lr': trial.suggest_float("lr", -10, 10)}

    def _update_model(self, params):
        self.trainner.update(params)

    def __call__(self, trial):
        # step1 sample hyperparams
        params = self._get_hyperparameters(trial)
        # step 2 update model
        self._update_model(params)
        # step 3 train
        acc = self.trainner.train()
        # step 4 report validation acc to controller
        # optuna minizes
        return  - acc

if __name__ == "__main__":
    model = trainner(config={'lr':0.1, 'patch_size':[96,96,96]})
    optuna_wrapper = OPTUNA_wrapper(trainner=model)
    study = optuna.create_study()
    study.optimize(optuna_wrapper, n_trials=100)
    print(f"Best value: {study.best_value} (params: {study.best_params})\n")
