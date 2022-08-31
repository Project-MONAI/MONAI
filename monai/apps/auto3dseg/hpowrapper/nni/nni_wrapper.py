import nni
import pdb
import sys
import yaml
sys.path.append('..')
from hpo_wrapper import HPO_wrapper

class NNI_wrapper(HPO_wrapper):
    def __init__(self, algo_name, task_folder, task_module, **kwargs):
        super().__init__(algo_name, task_folder, task_module, **kwargs)
    
    def _get_hyperparameters(self):
        return nni.get_next_parameter()
        
    def _update_algo(self, params):
        self.algo.update(params)

    def __call__(self):
        # step1 sample hyperparams
        params = self._get_hyperparameters()
        # step 2 update model
        self._update_algo(params) # save bundle
        # step 3 train
        acc = self.algo.train()
        # step 4 report validation acc to controller
        nni.report_final_result(acc)


def main():
    nni_wrapper = NNI_wrapper(algo_name='dummy', 
                              task_folder='/home/yufan/Projects/MONAI/monai/apps/auto3dseg/Task05_Prostate',
                              task_module='monai.apps.auto3dseg.Task05_Prostate.dummy.scripts.train')
    nni_wrapper()
    
if __name__ == "__main__":
    main()