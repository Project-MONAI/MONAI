import optuna
import sys
sys.path.append('..')
from hpo_wrapper import HPO_wrapper

class OPTUNA_wrapper(HPO_wrapper):
    def __init__(self, algo_name, task_folder, task_module, **kwargs):
        super().__init__(algo_name, task_folder, task_module, **kwargs)

    def _get_hyperparameters(self, trial):
        return {'lr': trial.suggest_float("lr", -10, 10)}

    def _update_algo(self, params):
        self.algo.update(params)

    def __call__(self, trial):
        # step1 sample hyperparams
        params = self._get_hyperparameters(trial)
        # step 2 update model
        self._update_algo(params)
        # step 3 train
        acc = self.algo.train(self.task_module)
        # step 4 report validation acc to controller
        # optuna minizes
        return  - acc

if __name__ == "__main__":
    optuna_wrapper = OPTUNA_wrapper(algo_name='dummy',
                                    task_folder='/home/yufan/Projects/MONAI/monai/apps/auto3dseg/Task05_Prostate',
                                    task_module='monai.apps.auto3dseg.Task05_Prostate_NNI_Trial100.dummy.scripts.train')
    study = optuna.create_study()
    study.optimize(optuna_wrapper, n_trials=100)
    print(f"Best value: {study.best_value} (params: {study.best_params})\n")
