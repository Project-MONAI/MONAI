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

from typing import Any

from monai.apps.auto3dseg import AutoRunner
from monai.apps.auto3dseg.utils import AZUREML_CONFIG_KEY, submit_auto3dseg_module_to_azureml_if_needed
from monai.apps.utils import get_logger
from monai.utils.module import optional_import
from monai.transforms import SaveImage

logger = get_logger(module_name=__name__)

nni, has_nni = optional_import("nni")
health_azure, has_health_azure = optional_import("health_azure")


class AzureMLAutoRunner(AutoRunner):
    """
    Subclass of AutoRunner that runs the training in AzureML instead of on local resources. Inputs are idnetical to
    those of AutoRunner, but the `input` argument must be a dictionary or input.yaml file containing the key
    `azureml_config` that contains the configuration for the AzureML run.

    """

    def __init__(
        self,
        input: dict[str, Any] | str | None = None,
        algos: dict | list | str | None = None,
        analyze: bool | None = None,
        algo_gen: bool | None = None,
        train: bool | None = None,
        training_params: dict[str, Any] | None = None,
        num_fold: int = 5,
        hpo: bool = False,
        hpo_backend: str = "nni",
        ensemble: bool = True,
        not_use_cache: bool = False,
        templates_path_or_url: str | None = None,
        **kwargs: Any,
    ):

        work_dir = "outputs"
        super().__init__(
            work_dir=work_dir,
            input=input,
            algos=algos,
            analyze=analyze,
            algo_gen=algo_gen,
            train=train,
            training_params=training_params,
            num_fold=num_fold,
            hpo=hpo,
            hpo_backend=hpo_backend,
            ensemble=ensemble,
            not_use_cache=not_use_cache,
            templates_path_or_url=templates_path_or_url,
            **kwargs,
        )

        run_info = submit_auto3dseg_module_to_azureml_if_needed(self.data_src_cfg[AZUREML_CONFIG_KEY])
        if run_info.input_datasets:
            self.dataroot = run_info.input_datasets[0]
            self.data_src_cfg["dataroot"] = str(self.dataroot)
            self._create_work_dir_and_data_src_cfg()
        else:
            self.dataroot = self.data_src_cfg["dataroot"]

    def _create_work_dir_and_data_src_cfg(self, data_src_cfg: dict[str, Any] | None = None) -> None:
        """
        Creates the work dir to be used by AutoRunner and exports the data source config to the specified filename if
        running in AzureML, do nothing otherwise.

        Args
            param data_src_cfg: dictionary containing the configuration for the AutoRunner, defaults to None
        """
        if health_azure.utils.is_running_in_azure_ml():
            super()._create_work_dir_and_data_src_cfg(data_src_cfg)
        else:
            pass

    def set_image_save_transform(self, kwargs) -> SaveImage | None:
        """
        Set the ensemble output transform if running in AzureML, otherwise do nothing.

        Args:
            kwargs: image writing parameters for the ensemble inference. The kwargs format follows SaveImage
                transform. For more information, check https://docs.monai.io/en/stable/transforms.html#saveimage .

        """
        if health_azure.utils.is_running_in_azure_ml():
            return super().set_image_save_transform(kwargs)
        else:
            pass

    def export_cache(self, **kwargs) -> None:
        """
        Export cache to the AzureML job working dir if running in AzureML, otherwise do nothing.
        """
        if health_azure.utils.is_running_in_azure_ml():
            return super().export_cache(**kwargs)
        else:
            pass
