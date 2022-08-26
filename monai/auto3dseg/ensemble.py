# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
import argparse
import subprocess

import monai.bundle
import torch
from monai.bundle import ConfigParser
from monai.engines import EnsembleEvaluator
from monai.utils import optional_import

KFold, _ = optional_import("sklearn.model_selection", name="KFold")
logger = logging.getLogger(__name__)


class Const:
    CONFIGS = ("train.json", "train.yaml")
    MULTI_GPU_CONFIGS = ("multi_gpu_train.json", "multi_gpu_train.yaml")
    INFERENCE_CONFIGS = ("inference.json", "inference.yaml")
    METADATA_JSON = "metadata.json"

    KEY_DEVICE = "device"
    KEY_BUNDLE_ROOT = "bundle_root"
    KEY_NETWORK = "network"
    KEY_NETWORK_DEF = "network_def"
    KEY_DATASET_DIR = "dataset_dir"
    KEY_TRAIN_TRAINER_MAX_EPOCHS = "train#trainer#max_epochs"
    KEY_TRAIN_DATASET_DATA = "train#dataset#data"
    KEY_VALIDATE_DATASET_DATA = "validate#dataset#data"
    KEY_INFERENCE_DATASET_DATA = "dataset#data"
    KEY_MODEL_PYTORCH = "validate#handlers#-1#key_metric_filename"
    KEY_INFERENCE_POSTPROCESSING = "postprocessing"


class EnsembleTrainTask():
    """
    To construct an n-fold training and ensemble infer on any dataset.
    Just specify the bundle root path and data root path.
    Bundle can be download from https://github.com/Project-MONAI/model-zoo/releases/tag/hosting_storage_v1
    Date root path also need a dataset.json which should be like:
        train datalist: [
            {
                "image": $image1_path,
                "label": $label1_path
            },
            ...
        ]
        test_datalist: [
            {
                "image": $image1_path
            },
            ...
        ]

    Args:
        path: bundle root path where your place the download bundle
    """
    def __init__(self, path):
        config_paths = [c for c in Const.CONFIGS if os.path.exists(os.path.join(path, "configs", c))]
        if not config_paths:
            logger.warning(f"Ignore {path} as there is no train config {Const.CONFIGS} exists")
            return

        self.bundle_path = path
        self.bundle_config_path = os.path.join(path, "configs", config_paths[0])

        self.bundle_config = ConfigParser()
        self.bundle_config.read_config(self.bundle_config_path)
        self.bundle_config.update({Const.KEY_BUNDLE_ROOT: self.bundle_path})

        self.bundle_metadata_path = os.path.join(path, "configs", Const.METADATA_JSON)

    def _partition_datalist(self, datalist, n_splits=5, shuffle=False):
        logger.info(f"Total Records in Dataset: {len(datalist)}")
        kfold = KFold(n_splits=n_splits, shuffle=shuffle)

        train_datalist, val_datalist = [], []
        for train_idx, valid_idx in kfold.split(datalist):
            train_datalist.append([datalist[i] for i in train_idx])
            val_datalist.append([datalist[i] for i in valid_idx])

        logger.info(f"Total Records for Training: {len(train_datalist[0])}")
        logger.info(f"Total Records for Validation: {len(val_datalist[0])}")
        return train_datalist, val_datalist

    def _device(self, str):
        return torch.device(str if torch.cuda.is_available() else "cpu")

    def ensemble_inference(self, device, test_datalist, ensemble='Mean'):
        inference_config_paths = [c for c in Const.INFERENCE_CONFIGS if os.path.exists(os.path.join(self.bundle_path, "configs", c))]
        if not inference_config_paths:
            logger.warning(f"Ignore {self.bundle_path} as there is no inference config {Const.INFERENCE_CONFIGS} exists")
            return

        logger.info(f"Total Records in Test Dataset: {len(test_datalist)}")

        bundle_inference_config_path = os.path.join(self.bundle_path, "configs", inference_config_paths[0])
        bundle_inference_config = ConfigParser()
        bundle_inference_config.read_config(bundle_inference_config_path)
        bundle_inference_config.update({Const.KEY_BUNDLE_ROOT: self.bundle_path})
        bundle_inference_config.update({Const.KEY_INFERENCE_DATASET_DATA: test_datalist})

        # update postprocessing with mean ensemble or vote ensemble
        post_tranform = bundle_inference_config.config['postprocessing']
        ensemble_tranform = {
            "_target_": f"{ensemble}Ensembled",
            "keys": ["pred", "pred", "pred", "pred", "pred"],
            "output_key": "pred"
        }
        if ensemble == 'Mean':
            post_tranform["transforms"].insert(0, ensemble_tranform)
        elif ensemble == 'Vote':
            post_tranform["transforms"].insert(-1, ensemble_tranform)
        else:
            raise NotImplementedError
        bundle_inference_config.update({Const.KEY_INFERENCE_POSTPROCESSING: post_tranform})

        # update network weights
        _networks = [bundle_inference_config.get_parsed_content("network")] * 5
        networks = []
        for i, _network in enumerate(_networks):
            _network.load_state_dict(torch.load(self.bundle_path+f"/models/model{i}.pt"))
            networks.append(_network)

        evaluator = EnsembleEvaluator(
            device=device,
            val_data_loader=bundle_inference_config.get_parsed_content("dataloader"),
            pred_keys=["pred", "pred", "pred", "pred", "pred"],
            networks=networks,
            inferer=bundle_inference_config.get_parsed_content("inferer"),
            postprocessing=bundle_inference_config.get_parsed_content("postprocessing"),
        )
        evaluator.run()
        logger.info(f"Inference Finished....")

    def __call__(self, args, datalist, test_datalist=None):
        dataset_dir = args.dataset_dir
        if dataset_dir is None:
            logger.warning(f"Ignore dataset dir as there is no dataset dir exists")
            return

        train_ds, val_ds = self._partition_datalist(datalist, n_splits=args.n_splits)
        fold = 0
        # for _train_ds, _val_ds in zip(train_ds, val_ds):
        #     model_pytorch = f'model{fold}.pt'
        #     max_epochs = args.epochs
        #     multi_gpu = args.multi_gpu
        #     multi_gpu = multi_gpu if torch.cuda.device_count() > 1 else False

        #     gpus = args.gpus
        #     gpus = list(range(torch.cuda.device_count())) if gpus == "all" else [int(g) for g in gpus.split(",")]
        #     logger.info(f"Using Multi GPU: {multi_gpu}; GPUS: {gpus}")
        #     logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

        #     device = self._device(args.device)
        #     logger.info(f"Using device: {device}")

        #     overrides = {
        #         Const.KEY_BUNDLE_ROOT: self.bundle_path,
        #         Const.KEY_TRAIN_TRAINER_MAX_EPOCHS: max_epochs,
        #         Const.KEY_TRAIN_DATASET_DATA: _train_ds,
        #         Const.KEY_VALIDATE_DATASET_DATA: _val_ds,
        #         Const.KEY_DATASET_DIR: dataset_dir,
        #         Const.KEY_MODEL_PYTORCH: model_pytorch,
        #         Const.KEY_DEVICE: device,
        #     }

        #     if multi_gpu:
        #         config_paths = [
        #             c for c in Const.MULTI_GPU_CONFIGS if os.path.exists(os.path.join(self.bundle_path, "configs", c))
        #         ]
        #         if not config_paths:
        #             logger.warning(f"Ignore Multi-GPU Training; No multi-gpu train config {Const.MULTI_GPU_CONFIGS} exists")
        #             return

        #         train_path = os.path.join(self.bundle_path, "configs", f"train_multigpu_fold{fold}.json")
        #         multi_gpu_train_path = os.path.join(self.bundle_path, "configs", config_paths[0])
        #         logging_file = os.path.join(self.bundle_path, "configs", "logging.conf")
        #         for k, v in overrides.items():
        #             if k != Const.KEY_DEVICE:
        #                 self.bundle_config.set(v, k)
        #         ConfigParser.export_config_file(self.bundle_config.config, train_path, indent=2)

        #         env = os.environ.copy()
        #         env["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in gpus])
        #         logger.info(f"Using CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
        #         cmd = [
        #             "torchrun",
        #             "--standalone",
        #             "--nnodes=1",
        #             f"--nproc_per_node={len(gpus)}",
        #             "-m",
        #             "monai.bundle",
        #             "run",
        #             "training",
        #             "--meta_file",
        #             self.bundle_metadata_path,
        #             "--config_file",
        #             f"['{train_path}','{multi_gpu_train_path}']",
        #             "--logging_file",
        #             logging_file,
        #         ]
        #         self.run_command(cmd, env)
        #     else:
        #         monai.bundle.run(
        #             "training",
        #             meta_file=self.bundle_metadata_path,
        #             config_file=self.bundle_config_path,
        #             **overrides,
        #         )
        #     fold += 1

        #     logger.info(f"Fold{fold} Training Finished....")

        if test_datalist is not None:
            device = self._device(args.device)
            self.ensemble_inference(device, test_datalist, ensemble=args.ensemble)

    def run_command(self, cmd, env):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, env=env)
        while process.poll() is None:
            line = process.stdout.readline()
            line = line.rstrip()
            if line:
                print(line, flush=True)

        logger.info(f"Return code: {process.returncode}")
        process.stdout.close()


if __name__ == '__main__':
    """
    Usage
        first download a bundle from model-zoo to somewhere as your bundle_root path
        split your data into train and test datalist
            train datalist: [
                {
                    "image": $image1_path
                    "label": $label1_path
                },
                {
                    "image": $image2_path,
                    "label": $label2_path
                },
                ...
            ]
            test_datalist: [
                {
                    "image": $image1_path
                },
                ...
            ]
        python easy_integrate_bundle.py --bundle_root $bundle_root_path --dataset_dir $data_root_path
    """
    parser = argparse.ArgumentParser(description="Run an ensemble train task using bundle.")

    parser.add_argument(
        "--ensemble", default="Mean", choices=["Mean", "Vote"], type=str, help="way of ensemble"
    )
    parser.add_argument("--bundle_root", default="", type=str, help="root bundle dir")
    parser.add_argument("--dataset_dir", default="", type=str, help="root data dir")
    parser.add_argument("--epochs", default=6, type=int, help="max epochs")
    parser.add_argument("--n_splits", default=5, type=int, help="n fold split")
    parser.add_argument("--multi_gpu", default=False, type=bool, help="whether use multigpu")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--gpus", default="all", type=str, help="which gpu to use")

    args = parser.parse_args()
    gpus = list(range(torch.cuda.device_count())) if args.gpus == "all" else [int(g) for g in args.gpus.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpus)
    datalist_path = args.dataset_dir+'/dataset.json'
    with open(datalist_path) as fp:
        datalist = json.load(fp)


    train_datalist = [{"image": d["image"].replace('./', f'{args.dataset_dir}/'), "label": d["label"].replace('./', f'{args.dataset_dir}/')} for d in datalist['training'] if d]
    test_datalist = [{"image": d.replace('./', f'{args.dataset_dir}/')} for d in datalist['test'] if d]
    traintask = EnsembleTrainTask(args.bundle_root)
    traintask(args, train_datalist, test_datalist)