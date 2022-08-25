import nni
import pdb
import sys
import yaml
sys.path.append('..')
from dummy_trainner import trainner
from hpo_wrapper import HPO_wrapper

class NNI_wrapper(HPO_wrapper):
    def __init__(self, trainner=None, **kwargs):
        super().__init__(trainner, **kwargs)
    
    def _get_hyperparameters(self):
        return nni.get_next_parameter()
        
    def _update_model(self, params):
        self.trainner.update(params)

    def run(self):
        # step1 sample hyperparams
        params = self._get_hyperparameters()
        # step 2 update model
        self._update_model(params)
        # step 3 train
        acc = self.trainner.train()
        # step 4 report validation acc to controller
        nni.report_final_result(acc)



def main():
    model = trainner(config={'lr':0.1})
    nni_wrapper = NNI_wrapper(trainner=model)
    nni_wrapper.run()

if __name__ == "__main__":
    main()