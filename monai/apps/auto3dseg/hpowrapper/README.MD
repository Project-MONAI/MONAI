A wrapper for general HPO packages
hpo_wrapper.py defines a HPO_Wrapper class
For each package (NNI, Optuna, Raytune e.t.c) should inherit this class
to create a wrapper.

For running NNI, 
```
cd nni;
bash nni_wrapper.sh
```

For running NNI, 
```
cd optuna;
bash optuna_wrapper.sh
```