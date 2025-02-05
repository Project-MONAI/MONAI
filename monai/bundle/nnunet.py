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
from __future__ import annotations

import os

import numpy as np
import torch
from torch._dynamo import OptimizedModule
from torch.backends import cudnn

from pathlib import Path
import shutil
from monai.data.meta_tensor import MetaTensor
from monai.utils import optional_import

join, _ = optional_import("batchgenerators.utilities.file_and_folder_operations", name="join")
load_json, _ = optional_import("batchgenerators.utilities.file_and_folder_operations", name="load_json")

__all__ = ["get_nnunet_trainer", "get_nnunet_monai_predictor", "nnUNetMONAIModelWrapper"]


def get_nnunet_trainer(
    dataset_name_or_id,
    configuration,
    fold,
    trainer_class_name="nnUNetTrainer",
    plans_identifier="nnUNetPlans",
    pretrained_weights=None,
    num_gpus=1,
    use_compressed_data=False,
    export_validation_probabilities=False,
    continue_training=False,
    only_run_validation=False,
    disable_checkpointing=False,
    val_with_best=False,
    device=torch.device("cuda"),
    pretrained_model=None,
):
    """
    Get the nnUNet trainer instance based on the provided configuration.
    The returned nnUNet trainer can be used to initialize the SupervisedTrainer for training, including the network,
    optimizer, loss function, DataLoader, etc.

    ```python
    from monai.apps import SupervisedTrainer
    from monai.bundle.nnunet import get_nnunet_trainer

    dataset_name_or_id = 'Task101_PROSTATE'
    fold = 0
    configuration = '3d_fullres'
    nnunet_trainer = get_nnunet_trainer(dataset_name_or_id, configuration, fold)

    trainer = SupervisedTrainer(
        device=nnunet_trainer.device,
        max_epochs=nnunet_trainer.num_epochs,
        train_data_loader=nnunet_trainer.dataloader_train,
        network=nnunet_trainer.network,
        optimizer=nnunet_trainer.optimizer,
        loss_function=nnunet_trainer.loss_function,
        epoch_length=nnunet_trainer.num_iterations_per_epoch,

    ```

    Parameters
    ----------
    dataset_name_or_id : Union[str, int]
        The name or ID of the dataset to be used.
    configuration : str
        The configuration name for the training.
    fold : Union[int, str]
        The fold number or 'all' for cross-validation.
    trainer_class_name : str, optional
        The class name of the trainer to be used. Default is 'nnUNetTrainer'.
    plans_identifier : str, optional
        Identifier for the plans to be used. Default is 'nnUNetPlans'.
    pretrained_weights : str, optional
        Path to the pretrained weights file.
    num_gpus : int, optional
        Number of GPUs to be used. Default is 1.
    use_compressed_data : bool, optional
        Whether to use compressed data. Default is False.
    export_validation_probabilities : bool, optional
        Whether to export validation probabilities. Default is False.
    continue_training : bool, optional
        Whether to continue training from a checkpoint. Default is False.
    only_run_validation : bool, optional
        Whether to only run validation. Default is False.
    disable_checkpointing : bool, optional
        Whether to disable checkpointing. Default is False.
    val_with_best : bool, optional
        Whether to validate with the best model. Default is False.
    device : torch.device, optional
        The device to be used for training. Default is 'cuda'.
    pretrained_model : str, optional
        Path to the pretrained model file.
    Returns
    -------
    nnunet_trainer
        The nnUNet trainer instance.
    """
    # From nnUNet/nnunetv2/run/run_training.py#run_training
    if isinstance(fold, str):
        if fold != "all":
            try:
                fold = int(fold)
            except ValueError as e:
                print(
                    f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!'
                )
                raise e

    if int(num_gpus) > 1:
        ...  # Disable for now
    else:
        from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint

        nnunet_trainer = get_trainer_from_args(
            str(dataset_name_or_id),
            configuration,
            fold,
            trainer_class_name,
            plans_identifier,
            use_compressed_data,
            device=device,
        )
        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (continue_training and only_run_validation), "Cannot set --c and --val flag at the same time. Dummy."

        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)
        nnunet_trainer.on_train_start()  # Added to Initialize Trainer
        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if pretrained_model is not None:
            state_dict = torch.load(pretrained_model)
            if "network_weights" in state_dict:
                nnunet_trainer.network._orig_mod.load_state_dict(state_dict["network_weights"])
        return nnunet_trainer


class nnUNetMONAIModelWrapper(torch.nn.Module):
    """
    A wrapper class for nnUNet model integration with MONAI framework.
    The wrapper can be use to integrate the nnUNet Bundle within MONAI framework for inference.

    Parameters
    ----------
    predictor : object
        The nnUNet predictor object used for inference.
    model_folder : str
        The folder path where the model and related files are stored.
    model_name : str, optional
        The name of the model file, by default "model.pt".
    Attributes
    ----------
    predictor : object
        The predictor object used for inference.
    network_weights : torch.nn.Module
        The network weights of the model.
    Methods
    -------
    forward(x)
        Perform forward pass and prediction on the input data.
    Notes
    -----
    This class integrates nnUNet model with MONAI framework by loading necessary configurations,
    restoring network architecture, and setting up the predictor for inference.
    """

    def __init__(self, predictor, model_folder, model_name="model.pt"):
        super().__init__()
        self.predictor = predictor

        model_training_output_dir = model_folder
        use_folds = "0"

        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        ## Block Added from nnUNet/nnunetv2/inference/predict_from_raw_data.py#nnUNetPredictor
        dataset_json = load_json(join(model_training_output_dir, "dataset.json"))
        plans = load_json(join(model_training_output_dir, "plans.json"))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != "all" else f
            checkpoint = torch.load(
                join(model_training_output_dir, "nnunet_checkpoint.pth"), map_location=torch.device("cpu")
            )
            monai_checkpoint = torch.load(join(model_training_output_dir, model_name), map_location=torch.device("cpu"))
            if i == 0:
                trainer_name = checkpoint["trainer_name"]
                configuration_name = checkpoint["init_args"]["configuration"]
                inference_allowed_mirroring_axes = (
                    checkpoint["inference_allowed_mirroring_axes"]
                    if "inference_allowed_mirroring_axes" in checkpoint.keys()
                    else None
                )

            parameters.append(monai_checkpoint["network_weights"])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        import nnunetv2
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), trainer_name, "nnunetv2.training.nnUNetTrainer"
        )
        if trainer_class is None:
            raise RuntimeError(
                f"Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. "
                f"Please place it there (in any .py file)!"
            )
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False,
        )

        predictor.plans_manager = plans_manager
        predictor.configuration_manager = configuration_manager
        predictor.list_of_parameters = parameters
        predictor.network = network
        predictor.dataset_json = dataset_json
        predictor.trainer_name = trainer_name
        predictor.allowed_mirroring_axes = inference_allowed_mirroring_axes
        predictor.label_manager = plans_manager.get_label_manager(dataset_json)
        if (
            ("nnUNet_compile" in os.environ.keys())
            and (os.environ["nnUNet_compile"].lower() in ("true", "1", "t"))
            and not isinstance(predictor.network, OptimizedModule)
        ):
            print("Using torch.compile")
            predictor.network = torch.compile(self.network)
        ## End Block
        self.network_weights = self.predictor.network

    def forward(self, x):
        if type(x) is tuple:  # if batch is decollated (list of tensors)
            input_files = [img.meta["filename_or_obj"][0] for img in x]
        else: # if batch is collated
            input_files = x.meta["filename_or_obj"]
            if type(input_files) is str:
                input_files = [input_files]
        
        # input_files should be a list of file paths, one per modality
        prediction_output = self.predictor.predict_from_files(
            [input_files],
            None,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
        # prediction_output is a list of numpy arrays, with dimensions (H, W, D), output from ArgMax
        
        out_tensors = []
        for out in prediction_output: # Add batch and channel dimensions
            out_tensors.append(torch.from_numpy(np.expand_dims(np.expand_dims(out, 0), 0)))
        out_tensor = torch.cat(out_tensors, 0) # Concatenate along batch dimension

        if type(x) is tuple:
            return MetaTensor(out_tensor, meta=x[0].meta)
        else:
            return MetaTensor(out_tensor, meta=x.meta)


def get_nnunet_monai_predictor(model_folder, model_name="model.pt"):
    """
    Initializes and returns a nnUNetMONAIModelWrapper with a nnUNetPredictor.
    The model folder should contain the following files, created during training:
    - dataset.json: from the nnUNet results folder.
    - plans.json: from the nnUNet results folder.
    - nnunet_checkpoint.pth: The nnUNet checkpoint file, containing the nnUNet training configuration
    (`init_kwargs`, `trainer_name`, `inference_allowed_mirroring_axes`).
    - model.pt: The checkpoint file containing the model weights.

    The returned wrapper object can be used for inference with MONAI framework:
    ```python
    from monai.bundle.nnunet import get_nnunet_monai_predictor

    model_folder = 'path/to/monai_bundle/model'
    model_name = 'model.pt'
    wrapper = get_nnunet_monai_predictor(model_folder, model_name)

    # Perform inference
    input_data = ...
    output = wrapper(input_data)

    ```

    Parameters
    ----------
    model_folder : str
        The folder where the model is stored.
    model_name : str, optional
        The name of the model file, by default "model.pt".

    Returns
    -------
    nnUNetMONAIModelWrapper
        A wrapper object that contains the nnUNetPredictor and the loaded model.
    """

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    # initializes the network architecture, loads the checkpoint
    wrapper = nnUNetMONAIModelWrapper(predictor, model_folder, model_name)
    return wrapper


def convert_nnunet_to_monai_bundle(nnunet_config, bundle_root_folder, fold=0):
    """
    Convert nnUNet model checkpoints and configuration to MONAI bundle format.

    Parameters
    ----------
    nnunet_config : dict
        Configuration dictionary for nnUNet, containing keys such as 'dataset_name_or_id', 'nnunet_configuration',
        'nnunet_trainer', and 'nnunet_plans'.
    bundle_root_folder : str
        Root folder where the MONAI bundle will be saved.
    fold : int, optional
        Fold number of the nnUNet model to be converted, by default 0.

    Returns
    -------
    None
    """

    nnunet_trainer = "nnUNetTrainer"
    nnunet_plans = "nnUNetPlans"
    nnunet_configuration = "3d_fullres"

    if "nnunet_trainer" in nnunet_config:
        nnunet_trainer = nnunet_config["nnunet_trainer"]

    if "nnunet_plans" in nnunet_config:
        nnunet_plans = nnunet_config["nnunet_plans"]

    if "nnunet_configuration" in nnunet_config:
        nnunet_configuration = nnunet_config["nnunet_configuration"]

    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


    dataset_name = maybe_convert_to_dataset_name(nnunet_config["dataset_name_or_id"])
    nnunet_model_folder = Path(os.environ["nnUNet_results"]).joinpath(
        dataset_name,
        f"{nnunet_trainer}__{nnunet_plans}__{nnunet_configuration}")
    
    nnunet_checkpoint_final = torch.load(Path(nnunet_model_folder).joinpath(f"fold_{fold}","checkpoint_final.pth"))
    nnunet_checkpoint_best = torch.load(Path(nnunet_model_folder).joinpath(f"fold_{fold}","checkpoint_best.pth"))

    nnunet_checkpoint = {}
    nnunet_checkpoint['inference_allowed_mirroring_axes'] = nnunet_checkpoint_final['inference_allowed_mirroring_axes']
    nnunet_checkpoint['init_args'] = nnunet_checkpoint_final['init_args']
    nnunet_checkpoint['trainer_name'] = nnunet_checkpoint_final['trainer_name']

    torch.save(nnunet_checkpoint, Path(bundle_root_folder).joinpath("models","nnunet_checkpoint.pth"))

    monai_last_checkpoint = {}
    monai_last_checkpoint['network_weights'] = nnunet_checkpoint_final['network_weights']
    torch.save(monai_last_checkpoint, Path(bundle_root_folder).joinpath("models","model.pt"))

    monai_best_checkpoint = {}
    monai_best_checkpoint['network_weights'] = nnunet_checkpoint_best['network_weights']
    torch.save(monai_best_checkpoint, Path(bundle_root_folder).joinpath("models","best_model.pt"))

    shutil.copy(Path(nnunet_model_folder).joinpath("plans.json"),Path(bundle_root_folder).joinpath("models","plans.json"))
    shutil.copy(Path(nnunet_model_folder).joinpath("dataset.json"),Path(bundle_root_folder).joinpath("models","dataset.json"))