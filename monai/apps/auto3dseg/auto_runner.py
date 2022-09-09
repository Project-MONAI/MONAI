import os

import torch

from monai.apps.auto3dseg import DataAnalyzer, BundleGen, AlgoEnsembleBuilder, AlgoEnsembleBestN
from monai.apps.utils import get_logger

from monai.bundle import ConfigParser
from monai.utils.enums import AlgoEnsembleKeys

logger = get_logger(module_name=__name__)


class AutoRunner:
    """
    Auto3Dseg interface for minimal usage

    Args:
        data_src_cfg_name: path to a configuration file (yaml) that contains datalist, dataroot, and other params.
                The config will be in a form of {"modality": "ct", "datalist": "path_to_json_datalist", "dataroot":
                "path_dir_data"}
        work_dir: working directory to save the intermediate results
        analyze: on/off switch for data analyzer
        algo_gen: on/off switch for algoGen
        train: on/off switch for sequential schedule of model training
        no_cache: if no_cache is True, it will reset the status and not use any previous results

    Examples:

        ..code-block:: python
            work_dir = "./work_dir"
            filename = "path_to_data_cfg"
            runner = AutoRunner(data_src_cfg_filename, work_dir)
            runner.run()

        ..code-block:: python
            work_dir = "./work_dir"
            filename = "path_to_data_cfg"
            runner = AutoRunner(data_src_cfg_filename, work_dir)
            train_param = {
                "CUDA_VISIBLE_DEVICES": [0],
                "num_iterations": 8,
                "num_iterations_per_validation": 4,
                "num_images_per_batch": 2,
                "num_epochs": 2,
            }
            runner.set_training_params(train_param)  # 2 epochs
            runner.run()

    Notes:
        Expected results in the work_dir as below.
            .
            ├── algorithm_templates
            │   ├── dints
            │   ├── segresnet
            │   ├── segresnet2d
            │   ├── swinunetr
            │   └── unet
            ├── dints_0
            │   ├── configs
            │   ├── model_fold0
            │   └── scripts
            ├── dints_1
            │   ├── configs
            │   ├── model_fold1
            │   └── scripts
            ├── dints_2
            │   ├── configs
            │   ├── model_fold2
            │   ├── prediction_testing
            │   └── scripts
            ├── segresnet_0
            │   ├── configs
            │   ├── model_fold0
            │   └── scripts
            ├── segresnet_1
            │   ├── configs
            │   ├── model_fold1
            │   └── scripts
            ├── segresnet_2
            │   ├── configs
            │   ├── model_fold2
            │   └── scripts
            ├── segresnet2d_0
            │   ├── configs
            │   ├── model_fold0
            │   └── scripts
            ├── segresnet2d_1
            │   ├── configs
            │   ├── model_fold1
            │   └── scripts
            ├── segresnet2d_2
            │   ├── configs
            │   ├── model_fold2
            │   └── scripts
            ├── swinunetr_0
            │   ├── configs
            │   ├── model_fold0
            │   └── scripts
            ├── swinunetr_1
            │   ├── configs
            │   ├── model_fold1
            │   └── scripts
            └── swinunetr_2
                ├── configs
                ├── model_fold2
                └── scripts
    """
    def __init__(
        self,
        data_src_cfg_name,
        work_dir: str = '.',
        analyze: bool = True,
        algo_gen: bool = True,
        train: bool = True,
        ensemble: bool = True,
        no_cache: bool = False,
    ):
        self.data_src_cfg_filename = data_src_cfg_name

        cfg = ConfigParser.load_config_file(data_src_cfg_name)
        self.dataroot = cfg["dataroot"]
        self.datalist_filename = cfg["datalist"]

        if not os.path.isdir(work_dir):
            logger.info(f"{work_dir} does not exists. Creating...")
            os.makedirs(work_dir)
            logger.info(f"{work_dir} created")

        self.work_dir = os.path.abspath(work_dir)

        self.analyze = analyze
        self.algo_gen = algo_gen
        self.train = train
        self.ensemble = ensemble
        self.no_cache = no_cache

        # intermediate
        self.set_datastats_filename()
        self.set_training_params()
        self.set_num_fold()

        # other algorithm parameters
        self.n_best = 1

    def set_datastats_filename(self, datastats_filename: str = 'datastats.yaml'):
        self.datastats_filename = datastats_filename

    def get_datastats_filename(self):
        return self.datastats_filename

    def set_num_fold(self, num_fold=5):
        """set number of cross validation folds"""
        self.num_fold = num_fold

    def set_training_params(self, params = None):
        if params is None:
            gpus = [_i for _i in range(torch.cuda.device_count())]
            self.train_params = {
                "CUDA_VISIBLE_DEVICES": gpus,
            }
        else:
            self.train_params = params


    def run(self):
        """
        Run the autorunner
        """
        ## data analysis
        if self.analyze:
            da = DataAnalyzer(self.datalist_filename, self.dataroot, output_path=self.get_datastats_filename())
            da.get_all_case_stats()

        ## algorithm generation
        if self.algo_gen:
            bundle_generator = BundleGen(
                algo_path=self.work_dir,
                data_stats_filename=self.get_datastats_filename(),
                data_src_cfg_name=self.data_src_cfg_filename,
            )

            bundle_generator.generate(self.work_dir, num_fold=self.num_fold )
            self.history = bundle_generator.get_history()

        ## model training
        if self.train:
            for i, record in enumerate(self.history):
                for name, algo in record.items():
                    algo.train(self.train_params)

        ## model ensemble
        if self.ensemble:
            builder = AlgoEnsembleBuilder(self.history, self.data_src_cfg_filename)
            builder.set_ensemble_method(AlgoEnsembleBestN(n_best=self.n_best))
            ensemble = builder.get_ensemble()
            pred = ensemble()
            print(f"ensemble picked the following best {self.n_best:d}:")
            for algo in ensemble.get_algo_ensemble():
                print(algo[AlgoEnsembleKeys.ID])
