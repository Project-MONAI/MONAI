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
import shutil
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.backends import cudnn

from monai.data.meta_tensor import MetaTensor
from monai.utils import optional_import

join, _ = optional_import("batchgenerators.utilities.file_and_folder_operations", name="join")
load_json, _ = optional_import("batchgenerators.utilities.file_and_folder_operations", name="load_json")

__all__ = [
    "get_nnunet_trainer",
    "get_nnunet_monai_predictor",
    "get_network_from_nnunet_plans",
    "convert_nnunet_to_monai_bundle",
    "convert_monai_bundle_to_nnunet",
    "ModelnnUNetWrapper",
]


def get_nnunet_trainer(
    dataset_name_or_id: Union[str, int],
    configuration: str,
    fold: Union[int, str],
    trainer_class_name: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetPlans",
    use_compressed_data: bool = False,
    continue_training: bool = False,
    only_run_validation: bool = False,
    disable_checkpointing: bool = False,
    device: str = "cuda",
    pretrained_model: Optional[str] = None,
) -> Any:  # type: ignore
    """
    Get the nnUNet trainer instance based on the provided configuration.
    The returned nnUNet trainer can be used to initialize the SupervisedTrainer for training, including the network,
    optimizer, loss function, DataLoader, etc.

    Example::

        from monai.apps import SupervisedTrainer
        from monai.bundle.nnunet import get_nnunet_trainer

        dataset_name_or_id = 'Task009_Spleen'
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
        )

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
        For a complete list of supported trainers, check:
        https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/training/nnUNetTrainer/variants
    plans_identifier : str, optional
        Identifier for the plans to be used. Default is 'nnUNetPlans'.
    use_compressed_data : bool, optional
        Whether to use compressed data. Default is False.
    continue_training : bool, optional
        Whether to continue training from a checkpoint. Default is False.
    only_run_validation : bool, optional
        Whether to only run validation. Default is False.
    disable_checkpointing : bool, optional
        Whether to disable checkpointing. Default is False.
    device : str, optional
        The device to be used for training. Default is 'cuda'.
    pretrained_model : Optional[str], optional
        Path to the pretrained model file.

    Returns
    -------
    nnunet_trainer : object
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

    from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint

    nnunet_trainer = get_trainer_from_args(
        str(dataset_name_or_id), configuration, fold, trainer_class_name, plans_identifier, device=torch.device(device)
    )
    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (continue_training and only_run_validation), "Cannot set --c and --val flag at the same time. Dummy."

    maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation)
    nnunet_trainer.on_train_start()  # Added to Initialize Trainer
    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if pretrained_model is not None:
        state_dict = torch.load(pretrained_model)
        if "network_weights" in state_dict:
            nnunet_trainer.network._orig_mod.load_state_dict(state_dict["network_weights"])
    return nnunet_trainer


class ModelnnUNetWrapper(torch.nn.Module):
    """
    A wrapper class for nnUNet model integration with MONAI framework.
    The wrapper can be use to integrate the nnUNet Bundle within MONAI framework for inference.

    Parameters
    ----------
    predictor : nnUNetPredictor
        The nnUNet predictor object used for inference.
    model_folder : Union[str, Path]
        The folder path where the model and related files are stored.
    model_name : str, optional
        The name of the model file, by default "model.pt".

    Attributes
    ----------
    predictor : nnUNetPredictor
        The nnUNet predictor object used for inference.
    network_weights : torch.nn.Module
        The network weights of the model.

    Notes
    -----
    This class integrates nnUNet model with MONAI framework by loading necessary configurations,
    restoring network architecture, and setting up the predictor for inference.
    """

    def __init__(self, predictor: object, model_folder: Union[str, Path], model_name: str = "model.pt"):  # type: ignore
        super().__init__()
        self.predictor = predictor

        model_training_output_dir = model_folder

        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        # Block Added from nnUNet/nnunetv2/inference/predict_from_raw_data.py#nnUNetPredictor
        dataset_json = load_json(join(Path(model_training_output_dir).parent, "dataset.json"))
        plans = load_json(join(Path(model_training_output_dir).parent, "plans.json"))
        plans_manager = PlansManager(plans)

        parameters = []

        checkpoint = torch.load(
            join(Path(model_training_output_dir).parent, "nnunet_checkpoint.pth"), map_location=torch.device("cpu")
        )
        trainer_name = checkpoint["trainer_name"]
        configuration_name = checkpoint["init_args"]["configuration"]
        inference_allowed_mirroring_axes = (
            checkpoint["inference_allowed_mirroring_axes"]
            if "inference_allowed_mirroring_axes" in checkpoint.keys()
            else None
        )
        if Path(model_training_output_dir).joinpath(model_name).is_file():
            monai_checkpoint = torch.load(join(model_training_output_dir, model_name), map_location=torch.device("cpu"))
            if "network_weights" in monai_checkpoint.keys():
                parameters.append(monai_checkpoint["network_weights"])
            else:
                parameters.append(monai_checkpoint)

        configuration_manager = plans_manager.get_configuration(configuration_name)
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

        predictor.plans_manager = plans_manager  # type: ignore
        predictor.configuration_manager = configuration_manager  # type: ignore
        predictor.list_of_parameters = parameters  # type: ignore
        predictor.network = network  # type: ignore
        predictor.dataset_json = dataset_json  # type: ignore
        predictor.trainer_name = trainer_name  # type: ignore
        predictor.allowed_mirroring_axes = inference_allowed_mirroring_axes  # type: ignore
        predictor.label_manager = plans_manager.get_label_manager(dataset_json)  # type: ignore

        self.network_weights = self.predictor.network  # type: ignore

    def forward(self, x: MetaTensor) -> MetaTensor:
        """
        Forward pass for the nnUNet model.

        Args:
            x (MetaTensor): Input tensor. If the input is a tuple,
                it is assumed to be a decollated batch (list of tensors). Otherwise, it is assumed to be a collated batch.

        Returns:
            MetaTensor: The output tensor with the same metadata as the input.

        Raises:
            TypeError: If the input is not a torch.Tensor or a tuple of MetaTensors.

        Notes:
            - If the input is a tuple, the filenames are extracted from the metadata of each tensor in the tuple.
            - If the input is a collated batch, the filenames are extracted from the metadata of the input tensor.
            - The filenames are used to generate predictions using the nnUNet predictor.
            - The predictions are converted to torch tensors, with added batch and channel dimensions.
            - The output tensor is concatenated along the batch dimension and returned as a MetaTensor with the same metadata.
        """
        if isinstance(x, MetaTensor):
            if "pixdim" in x.meta:
                properties_or_list_of_properties = {"spacing": x.meta["pixdim"][0][1:4].numpy().tolist()}
            elif "affine" in x.meta:
                spacing = [
                    abs(x.meta["affine"][0][0].item()),
                    abs(x.meta["affine"][1][1].item()),
                    abs(x.meta["affine"][2][2].item()),
                ]
                properties_or_list_of_properties = {"spacing": spacing}
            else:
                properties_or_list_of_properties = {"spacing": [1.0, 1.0, 1.0]}
        else:
            raise TypeError("Input must be a MetaTensor or a tuple of MetaTensors.")

        image_or_list_of_images = x.cpu().numpy()[0, :]

        # input_files should be a list of file paths, one per modality
        prediction_output = self.predictor.predict_from_list_of_npy_arrays(  # type: ignore
            image_or_list_of_images,
            None,
            properties_or_list_of_properties,
            truncated_ofname=None,
            save_probabilities=False,
            num_processes=2,
            num_processes_segmentation_export=2,
        )
        # prediction_output is a list of numpy arrays, with dimensions (H, W, D), output from ArgMax

        out_tensors = []
        for out in prediction_output:  # Add batch and channel dimensions
            out_tensors.append(torch.from_numpy(np.expand_dims(np.expand_dims(out, 0), 0)))
        out_tensor = torch.cat(out_tensors, 0)  # Concatenate along batch dimension

        return MetaTensor(out_tensor, meta=x.meta)


def get_nnunet_monai_predictor(model_folder: Union[str, Path], model_name: str = "model.pt") -> ModelnnUNetWrapper:
    """
    Initializes and returns a `nnUNetMONAIModelWrapper` containing the corresponding `nnUNetPredictor`.
    The model folder should contain the following files, created during training:

        - dataset.json: from the nnUNet results folder
        - plans.json: from the nnUNet results folder
        - nnunet_checkpoint.pth: The nnUNet checkpoint file, containing the nnUNet training configuration
        - model.pt: The checkpoint file containing the model weights.

    The returned wrapper object can be used for inference with MONAI framework:

    Example::

        from monai.bundle.nnunet import get_nnunet_monai_predictor

        model_folder = 'path/to/monai_bundle/model'
        model_name = 'model.pt'
        wrapper = get_nnunet_monai_predictor(model_folder, model_name)

        # Perform inference
        input_data = ...
        output = wrapper(input_data)


    Parameters
    ----------
    model_folder : Union[str, Path]
        The folder where the model is stored.
    model_name : str, optional
        The name of the model file, by default "model.pt".

    Returns
    -------
    ModelnnUNetWrapper
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
    wrapper = ModelnnUNetWrapper(predictor, model_folder, model_name)
    return wrapper


def convert_nnunet_to_monai_bundle(nnunet_config: dict, bundle_root_folder: str, fold: int = 0) -> None:
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
        dataset_name, f"{nnunet_trainer}__{nnunet_plans}__{nnunet_configuration}"
    )

    nnunet_checkpoint_final = torch.load(Path(nnunet_model_folder).joinpath(f"fold_{fold}", "checkpoint_final.pth"))
    nnunet_checkpoint_best = torch.load(Path(nnunet_model_folder).joinpath(f"fold_{fold}", "checkpoint_best.pth"))

    nnunet_checkpoint = {}
    nnunet_checkpoint["inference_allowed_mirroring_axes"] = nnunet_checkpoint_final["inference_allowed_mirroring_axes"]
    nnunet_checkpoint["init_args"] = nnunet_checkpoint_final["init_args"]
    nnunet_checkpoint["trainer_name"] = nnunet_checkpoint_final["trainer_name"]

    torch.save(nnunet_checkpoint, Path(bundle_root_folder).joinpath("models", "nnunet_checkpoint.pth"))

    Path(bundle_root_folder).joinpath("models", f"fold_{fold}").mkdir(parents=True, exist_ok=True)
    monai_last_checkpoint = {}
    monai_last_checkpoint["network_weights"] = nnunet_checkpoint_final["network_weights"]
    torch.save(monai_last_checkpoint, Path(bundle_root_folder).joinpath("models", f"fold_{fold}", "model.pt"))

    monai_best_checkpoint = {}
    monai_best_checkpoint["network_weights"] = nnunet_checkpoint_best["network_weights"]
    torch.save(monai_best_checkpoint, Path(bundle_root_folder).joinpath("models", f"fold_{fold}", "best_model.pt"))

    if not os.path.exists(os.path.join(bundle_root_folder, "models", "plans.json")):
        shutil.copy(
            Path(nnunet_model_folder).joinpath("plans.json"), Path(bundle_root_folder).joinpath("models", "plans.json")
        )

    if not os.path.exists(os.path.join(bundle_root_folder, "models", "dataset.json")):
        shutil.copy(
            Path(nnunet_model_folder).joinpath("dataset.json"),
            Path(bundle_root_folder).joinpath("models", "dataset.json"),
        )


def get_network_from_nnunet_plans(
    plans_file: str,
    dataset_file: str,
    configuration: str,
    model_ckpt: Optional[str] = None,
    model_key_in_ckpt: str = "model",
) -> Union[torch.nn.Module, Any]:
    """
    Load and initialize a nnUNet network based on nnUNet plans and configuration.

    Parameters
    ----------
    plans_file : str
        Path to the JSON file containing the nnUNet plans.
    dataset_file : str
        Path to the JSON file containing the dataset information.
    configuration : str
        The configuration name to be used from the plans.
    model_ckpt : Optional[str], optional
        Path to the model checkpoint file. If None, the network is returned without loading weights (default is None).
    model_key_in_ckpt : str, optional
        The key in the checkpoint file that contains the model state dictionary (default is "model").

    Returns
    -------
    network : torch.nn.Module
        The initialized neural network, with weights loaded if `model_ckpt` is provided.
    """
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans = load_json(plans_file)
    dataset_json = load_json(dataset_file)

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    label_manager = plans_manager.get_label_manager(dataset_json)

    enable_deep_supervision = True

    network = get_network_from_plans(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        label_manager.num_segmentation_heads,
        allow_init=True,
        deep_supervision=enable_deep_supervision,
    )

    if model_ckpt is None:
        return network
    else:
        state_dict = torch.load(model_ckpt)
        network.load_state_dict(state_dict[model_key_in_ckpt])
        return network


def convert_monai_bundle_to_nnunet(nnunet_config: dict, bundle_root_folder: str, fold: int = 0) -> None:
    """
    Convert a MONAI bundle to nnU-Net format.

    Parameters
    ----------
    nnunet_config : dict
        Configuration dictionary for nnU-Net. Expected keys are:
        - "dataset_name_or_id": str, name or ID of the dataset.
        - "nnunet_trainer": str, optional, name of the nnU-Net trainer (default is "nnUNetTrainer").
        - "nnunet_plans": str, optional, name of the nnU-Net plans (default is "nnUNetPlans").
    bundle_root_folder : str
        Path to the root folder of the MONAI bundle.
    fold : int, optional
        Fold number for cross-validation (default is 0).

    Returns
    -------
    None
    """
    from odict import odict

    nnunet_trainer: str = "nnUNetTrainer"
    nnunet_plans: str = "nnUNetPlans"

    if "nnunet_trainer" in nnunet_config:
        nnunet_trainer = nnunet_config["nnunet_trainer"]

    if "nnunet_plans" in nnunet_config:
        nnunet_plans = nnunet_config["nnunet_plans"]

    from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    def subfiles(
        folder: Union[str, Path], prefix: Optional[str] = None, suffix: Optional[str] = None, sort: bool = True
    ) -> list[str]:
        res = [
            i.name
            for i in Path(folder).iterdir()
            if i.is_file()
            and (prefix is None or i.name.startswith(prefix))
            and (suffix is None or i.name.endswith(suffix))
        ]
        if sort:
            res.sort()
        return res

    nnunet_model_folder: Path = Path(os.environ["nnUNet_results"]).joinpath(
        maybe_convert_to_dataset_name(nnunet_config["dataset_name_or_id"]),
        f"{nnunet_trainer}__{nnunet_plans}__3d_fullres",
    )

    nnunet_preprocess_model_folder: Path = Path(os.environ["nnUNet_preprocessed"]).joinpath(
        maybe_convert_to_dataset_name(nnunet_config["dataset_name_or_id"])
    )

    Path(nnunet_model_folder).joinpath(f"fold_{fold}").mkdir(parents=True, exist_ok=True)

    nnunet_checkpoint: dict = torch.load(f"{bundle_root_folder}/models/nnunet_checkpoint.pth")
    latest_checkpoints: list[str] = subfiles(
        Path(bundle_root_folder).joinpath("models", f"fold_{fold}"), prefix="checkpoint_epoch", sort=True
    )
    epochs: list[int] = []
    for latest_checkpoint in latest_checkpoints:
        epochs.append(int(latest_checkpoint[len("checkpoint_epoch=") : -len(".pt")]))

    epochs.sort()
    final_epoch: int = epochs[-1]
    monai_last_checkpoint: dict = torch.load(
        f"{bundle_root_folder}/models/fold_{fold}/checkpoint_epoch={final_epoch}.pt"
    )

    best_checkpoints: list[str] = subfiles(
        Path(bundle_root_folder).joinpath("models", f"fold_{fold}"), prefix="checkpoint_key_metric", sort=True
    )
    key_metrics: list[str] = []
    for best_checkpoint in best_checkpoints:
        key_metrics.append(str(best_checkpoint[len("checkpoint_key_metric=") : -len(".pt")]))

    key_metrics.sort()
    best_key_metric: str = key_metrics[-1]
    monai_best_checkpoint: dict = torch.load(
        f"{bundle_root_folder}/models/fold_{fold}/checkpoint_key_metric={best_key_metric}.pt"
    )

    nnunet_checkpoint["optimizer_state"] = monai_last_checkpoint["optimizer_state"]

    nnunet_checkpoint["network_weights"] = odict()

    for key in monai_last_checkpoint["network_weights"]:
        nnunet_checkpoint["network_weights"][key] = monai_last_checkpoint["network_weights"][key]

    nnunet_checkpoint["current_epoch"] = final_epoch
    nnunet_checkpoint["logging"] = nnUNetLogger().get_checkpoint()
    nnunet_checkpoint["_best_ema"] = 0
    nnunet_checkpoint["grad_scaler_state"] = None

    torch.save(nnunet_checkpoint, Path(nnunet_model_folder).joinpath(f"fold_{fold}", "checkpoint_final.pth"))

    nnunet_checkpoint["network_weights"] = odict()

    nnunet_checkpoint["optimizer_state"] = monai_best_checkpoint["optimizer_state"]

    for key in monai_best_checkpoint["network_weights"]:
        nnunet_checkpoint["network_weights"][key] = monai_best_checkpoint["network_weights"][key]

    torch.save(nnunet_checkpoint, Path(nnunet_model_folder).joinpath(f"fold_{fold}", "checkpoint_best.pth"))

    if not os.path.exists(os.path.join(nnunet_model_folder, "dataset.json")):
        shutil.copy(f"{bundle_root_folder}/models/dataset.json", nnunet_model_folder)
    if not os.path.exists(os.path.join(nnunet_model_folder, "plans.json")):
        shutil.copy(f"{bundle_root_folder}/models/plans.json", nnunet_model_folder)
    if not os.path.exists(os.path.join(nnunet_model_folder, "dataset_fingerprint.json")):
        shutil.copy(f"{nnunet_preprocess_model_folder}/dataset_fingerprint.json", nnunet_model_folder)
    if not os.path.exists(os.path.join(nnunet_model_folder, "nnunet_checkpoint.pth")):
        shutil.copy(f"{bundle_root_folder}/models/nnunet_checkpoint.pth", nnunet_model_folder)
