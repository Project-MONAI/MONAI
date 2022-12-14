# What's new in 1.1 🎉🎉

- Experiment Management for MONAI bundle


## Experiment Management for MONAI bundle

In this release, we support the experiment management for MONAI bundle. It can help to track the experiment process so that users can easily reproduce experiments that have been done before and can easily retrace the information like super parameters and metrics of previous experiments. The default experiment management tool is [MLFLow](https://mlflow.org/docs/latest/tracking.html). Users can enable the MLFlow tracking by simply adding the `--tracking "mlflow"` option at the end of training or inference command lines. By default, MLFlow will record relative configs, metrics, parameters, starting time and code version of the running experiment. For more details about it, please refer to this [tutorial](https://github.com/Project-MONAI/tutorials/blob/main/experiment_management/bundle_integrate_mlflow.ipynb). In addition, users can also define their experiment management environment by adding a handler referring to [MLFlowHandler](../../monai/handlers/mlflow_handler.py) and a tracking dictionary referring to the `tracking` parameter in the `run` method in [MONAI bundle](../../monai/bundle/scripts.py).