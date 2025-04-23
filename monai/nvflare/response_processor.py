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

from nvflare.apis.client import Client
from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.response_processor import ResponseProcessor


class nnUNetPrepareProcessor(ResponseProcessor):
    """
    A processor class for preparing nnUNet data in a federated learning context.

    Methods
    -------
    __init__():
        Initializes the nnUNetPrepareProcessor with an empty data dictionary.
    create_task_data(task_name: str, fl_ctx: FLContext) -> Shareable:
        Creates and returns a Shareable object for the given task name.
    process_client_response(client: Client, task_name: str, response: Shareable, fl_ctx: FLContext) -> bool:
        Processes the response from a client. Validates the response and updates the data dictionary if valid.
    final_process(fl_ctx: FLContext) -> bool:
        Finalizes the processing by setting the client data dictionary in the federated learning context.
    """

    def __init__(self):
        ResponseProcessor.__init__(self)
        self.data_dict = {}

    def create_task_data(self, task_name: str, fl_ctx: FLContext) -> Shareable:
        return Shareable()

    def process_client_response(self, client: Client, task_name: str, response: Shareable, fl_ctx: FLContext) -> bool:
        if not isinstance(response, Shareable):
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: " f"response must be Shareable but got {type(response)}",
            )
            return False

        try:
            dxo = from_shareable(response)

        except Exception:
            self.log_exception(fl_ctx, f"bad response from client {client.name}: " f"it does not contain DXO")
            return False

        if dxo.data_kind != DataKind.COLLECTION:
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: "
                f"data_kind should be DataKind.COLLECTION but got {dxo.data_kind}",
            )
            return False

        data_dict = dxo.data

        if not data_dict:
            self.log_error(fl_ctx, f"No dataset_dict found from client {client.name}")
            return False

        self.data_dict[client.name] = data_dict

        return True

    def final_process(self, fl_ctx: FLContext) -> bool:
        if not self.data_dict:
            self.log_error(fl_ctx, "no data_prepare_dict from clients")
            return False

        # must set sticky to True so other controllers can get it!
        fl_ctx.set_prop("client_data_dict", self.data_dict, private=True, sticky=True)
        return True


class nnUNetPackageReportProcessor(ResponseProcessor):
    """
    A processor for handling nnUNet package reports in a federated learning context.

    Attributes
    ----------
    package_report : dict
        A dictionary to store package reports from clients.

    Methods
    -------
    create_task_data(task_name: str, fl_ctx: FLContext) -> Shareable
        Creates task data for a given task name and federated learning context.
    process_client_response(client: Client, task_name: str, response: Shareable, fl_ctx: FLContext) -> bool
        Processes the response from a client for a given task name and federated learning context.
    final_process(fl_ctx: FLContext) -> bool
        Final processing step to handle the collected package reports.
    """

    def __init__(self):
        ResponseProcessor.__init__(self)
        self.package_report = {}

    def create_task_data(self, task_name: str, fl_ctx: FLContext) -> Shareable:
        return Shareable()

    def process_client_response(self, client: Client, task_name: str, response: Shareable, fl_ctx: FLContext) -> bool:
        if not isinstance(response, Shareable):
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: " f"response must be Shareable but got {type(response)}",
            )
            return False

        try:
            dxo = from_shareable(response)

        except Exception:
            self.log_exception(fl_ctx, f"bad response from client {client.name}: " f"it does not contain DXO")
            return False

        if dxo.data_kind != DataKind.COLLECTION:
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: "
                f"data_kind should be DataKind.COLLECTION but got {dxo.data_kind}",
            )
            return False

        package_report = dxo.data

        if not package_report:
            self.log_error(fl_ctx, f"No package_report found from client {client.name}")
            return False

        self.package_report[client.name] = package_report
        return True

    def final_process(self, fl_ctx: FLContext) -> bool:
        if not self.package_report:
            self.log_error(fl_ctx, "no plan_dict from client")
            return False

        # must set sticky to True so other controllers can get it!
        fl_ctx.set_prop("package_report", self.package_report, private=True, sticky=True)
        return True


class nnUNetPlanProcessor(ResponseProcessor):
    """
    nnUNetPlanProcessor is a class that processes responses from clients in a federated learning context.
    It inherits from the ResponseProcessor class and is responsible for handling and validating the
    responses, extracting the necessary data, and storing it for further use.

    Attributes
    ----------
    plan_dict : dict
        A dictionary to store the plan data received from clients.

    Methods
    -------
    create_task_data(task_name: str, fl_ctx: FLContext) -> Shareable
        Creates and returns a Shareable object for the given task name.
    process_client_response(client: Client, task_name: str, response: Shareable, fl_ctx: FLContext) -> bool
        Processes the response from a client, validates it, and stores the plan data if valid.
    final_process(fl_ctx: FLContext) -> bool
        Finalizes the processing by setting the plan data in the federated learning context.
    """

    def __init__(self):
        ResponseProcessor.__init__(self)
        self.plan_dict = {}

    def create_task_data(self, task_name: str, fl_ctx: FLContext) -> Shareable:
        return Shareable()

    def process_client_response(self, client: Client, task_name: str, response: Shareable, fl_ctx: FLContext) -> bool:
        if not isinstance(response, Shareable):
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: " f"response must be Shareable but got {type(response)}",
            )
            return False

        try:
            dxo = from_shareable(response)

        except Exception:
            self.log_exception(fl_ctx, f"bad response from client {client.name}: " f"it does not contain DXO")
            return False

        if dxo.data_kind != DataKind.COLLECTION:
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: "
                f"data_kind should be DataKind.COLLECTION but got {dxo.data_kind}",
            )
            return False

        plan_dict = dxo.data

        if not plan_dict:
            self.log_error(fl_ctx, f"No plan_dict found from client {client.name}")
            return False

        self.plan_dict[client.name] = plan_dict

        return True

    def final_process(self, fl_ctx: FLContext) -> bool:
        if not self.plan_dict:
            self.log_error(fl_ctx, "no plan_dict from client")
            return False

        # must set sticky to True so other controllers can get it!
        fl_ctx.set_prop("nnunet_plans", self.plan_dict, private=True, sticky=True)
        return True


class nnUNetTrainProcessor(ResponseProcessor):
    """
    A processor class for handling training responses in the nnUNet framework.

    Attributes
    ----------
    val_summary_dict : dict
        A dictionary to store validation summaries from clients.
    Methods
    -------
    create_task_data(task_name: str, fl_ctx: FLContext) -> Shareable
        Creates task data for a given task name and FLContext.
    process_client_response(client: Client, task_name: str, response: Shareable, fl_ctx: FLContext) -> bool
        Processes the response from a client for a given task name and FLContext.
    final_process(fl_ctx: FLContext) -> bool
        Final processing step to handle the collected validation summaries.
    """

    def __init__(self):
        ResponseProcessor.__init__(self)
        self.val_summary_dict = {}

    def create_task_data(self, task_name: str, fl_ctx: FLContext) -> Shareable:
        return Shareable()

    def process_client_response(self, client: Client, task_name: str, response: Shareable, fl_ctx: FLContext) -> bool:
        if not isinstance(response, Shareable):
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: " f"response must be Shareable but got {type(response)}",
            )
            return False

        try:
            dxo = from_shareable(response)

        except Exception:
            self.log_exception(fl_ctx, f"bad response from client {client.name}: " f"it does not contain DXO")
            return False

        if dxo.data_kind != DataKind.COLLECTION:
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: "
                f"data_kind should be DataKind.COLLECTION but got {dxo.data_kind}",
            )
            return False

        val_summary_dict = dxo.data

        if not val_summary_dict:
            self.log_error(fl_ctx, f"No val_summary_dict found from client {client.name}")
            return False

        self.val_summary_dict[client.name] = val_summary_dict

        return True

    def final_process(self, fl_ctx: FLContext) -> bool:
        if not self.val_summary_dict:
            self.log_error(fl_ctx, "no val_summary_dict from client")
            return False

        # must set sticky to True so other controllers can get it!
        fl_ctx.set_prop("val_summary_dict", self.val_summary_dict, private=True, sticky=True)
        return True


class nnUNetBundlePrepareProcessor(ResponseProcessor):
    """
    A processor class for preparing nnUNet bundles in a federated learning context.

    Methods
    -------
    __init__():
        Initializes the nnUNetBundlePrepareProcessor instance.
    create_task_data(task_name: str, fl_ctx: FLContext) -> Shareable:
        Creates task data for a given task name and federated learning context.
    process_client_response(client: Client, task_name: str, response: Shareable, fl_ctx: FLContext) -> bool:
        Processes the response from a client and validates it.
    final_process(fl_ctx: FLContext) -> bool:
        Final processing step after all client responses have been processed.
    """

    def __init__(self):
        ResponseProcessor.__init__(self)
        self.bundle_config = {}

    def create_task_data(self, task_name: str, fl_ctx: FLContext) -> Shareable:
        return Shareable()

    def process_client_response(self, client: Client, task_name: str, response: Shareable, fl_ctx: FLContext) -> bool:
        if not isinstance(response, Shareable):
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: " f"response must be Shareable but got {type(response)}",
            )
            return False

        try:
            dxo = from_shareable(response)

        except Exception:
            self.log_exception(fl_ctx, f"bad response from client {client.name}: " f"it does not contain DXO")
            return False

        if dxo.data_kind != DataKind.COLLECTION:
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: "
                f"data_kind should be DataKind.COLLECTION but got {dxo.data_kind}",
            )
            return False

        bundle_config = dxo.data

        if not bundle_config:
            self.log_error(fl_ctx, f"No bundle_config found from client {client.name}")
            return False

        self.bundle_config[client.name] = bundle_config

        return True

    def final_process(self, fl_ctx: FLContext) -> bool:
        if not self.bundle_config:
            self.log_error(fl_ctx, "no bundle_config from client")
            return False

        # must set sticky to True so other controllers can get it!
        fl_ctx.set_prop("bundle_config", self.bundle_config, private=True, sticky=True)
        return True
