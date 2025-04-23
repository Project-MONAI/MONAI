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

import json
import os.path

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.widgets.widget import Widget


class PrepareJsonGenerator(Widget):
    """
    A widget class to prepare and generate a JSON file containing data preparation configurations.

    Parameters
    ----------
    results_dir : str, optional
        The directory where the results will be stored (default is "prepare").
    json_file_name : str, optional
        The name of the JSON file to be generated (default is "data_dict.json").

    Methods
    -------
    handle_event(event_type: str, fl_ctx: FLContext)
        Handles events during the federated learning process. Clears the data preparation configuration
        at the start of a run and saves the configuration to a JSON file at the end of a run.
    """

    def __init__(self, results_dir="prepare", json_file_name="data_dict.json"):
        super(PrepareJsonGenerator, self).__init__()

        self._results_dir = results_dir
        self._data_prepare_config = {}
        self._json_file_name = json_file_name

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._data_prepare_config.clear()
        elif event_type == EventType.END_RUN:
            self._data_prepare_config = fl_ctx.get_prop("client_data_dict", None)
            run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_job_id())
            data_prepare_res_dir = os.path.join(run_dir, self._results_dir)
            if not os.path.exists(data_prepare_res_dir):
                os.makedirs(data_prepare_res_dir)

            res_file_path = os.path.join(data_prepare_res_dir, self._json_file_name)
            with open(res_file_path, "w") as f:
                json.dump(self._data_prepare_config, f)


class nnUNetPackageReportJsonGenerator(Widget):
    """
    A class to generate JSON reports for nnUNet package.

    Parameters
    ----------
    results_dir : str, optional
        Directory where the report will be saved (default is "package_report").
    json_file_name : str, optional
        Name of the JSON file to save the report (default is "package_report.json").

    Methods
    -------
    handle_event(event_type: str, fl_ctx: FLContext)
        Handles events to clear the report at the start of a run and save the report at the end of a run.
    """

    def __init__(self, results_dir="package_report", json_file_name="package_report.json"):
        super(nnUNetPackageReportJsonGenerator, self).__init__()

        self._results_dir = results_dir
        self._report = {}
        self._json_file_name = json_file_name

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._report.clear()
        elif event_type == EventType.END_RUN:
            datasets = fl_ctx.get_prop("package_report", None)
            run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_job_id())
            cross_val_res_dir = os.path.join(run_dir, self._results_dir)
            if not os.path.exists(cross_val_res_dir):
                os.makedirs(cross_val_res_dir)

            res_file_path = os.path.join(cross_val_res_dir, self._json_file_name)
            with open(res_file_path, "w") as f:
                json.dump(datasets, f)


class nnUNetPlansJsonGenerator(Widget):
    """
    A class to generate JSON files for nnUNet plans.

    Parameters
    ----------
    results_dir : str, optional
        Directory where the preprocessing results will be stored (default is "nnUNet_preprocessing").
    json_file_name : str, optional
        Name of the JSON file to be generated (default is "nnUNetPlans.json").

    Methods
    -------
    handle_event(event_type: str, fl_ctx: FLContext)
        Handles events during the federated learning process. Clears the nnUNet plans at the start of a run and saves
        the plans to a JSON file at the end of a run.
    """

    def __init__(self, results_dir="nnUNet_preprocessing", json_file_name="nnUNetPlans.json"):

        super(nnUNetPlansJsonGenerator, self).__init__()

        self._results_dir = results_dir
        self._nnUNetPlans = {}
        self._json_file_name = json_file_name

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._nnUNetPlans.clear()
        elif event_type == EventType.END_RUN:
            datasets = fl_ctx.get_prop("nnunet_plans", None)
            run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_job_id())
            cross_val_res_dir = os.path.join(run_dir, self._results_dir)
            if not os.path.exists(cross_val_res_dir):
                os.makedirs(cross_val_res_dir)

            res_file_path = os.path.join(cross_val_res_dir, self._json_file_name)
            with open(res_file_path, "w") as f:
                json.dump(datasets, f)


class nnUNetValSummaryJsonGenerator(Widget):
    """
    A widget to generate a JSON summary for nnUNet validation results.

    Parameters
    ----------
    results_dir : str, optional
        Directory where the nnUNet training results are stored (default is "nnUNet_train").
    json_file_name : str, optional
        Name of the JSON file to save the validation summary (default is "val_summary.json").

    Methods
    -------
    handle_event(event_type: str, fl_ctx: FLContext)
        Handles events during the federated learning process. Clears the nnUNet plans at the start of a run and saves
        the validation summary to a JSON file at the end of a run.
    """

    def __init__(self, results_dir="nnUNet_train", json_file_name="val_summary.json"):

        super(nnUNetValSummaryJsonGenerator, self).__init__()

        self._results_dir = results_dir
        self._nnUNetPlans = {}
        self._json_file_name = json_file_name

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._nnUNetPlans.clear()
        elif event_type == EventType.END_RUN:
            datasets = fl_ctx.get_prop("val_summary_dict", None)
            run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_job_id())
            cross_val_res_dir = os.path.join(run_dir, self._results_dir)
            if not os.path.exists(cross_val_res_dir):
                os.makedirs(cross_val_res_dir)

            res_file_path = os.path.join(cross_val_res_dir, self._json_file_name)
            with open(res_file_path, "w") as f:
                json.dump(datasets, f)


class nnUNetPrepareBundleJsonGenerator(Widget):
    """
    A class to generate a JSON configuration file for nnUNet bundle preparation.

    This class listens to specific events during a federated learning run and generates
    a JSON file containing the bundle configuration at the end of the run.

    Parameters
    ----------
    results_dir : str, optional
        The directory where the JSON configuration file will be saved. Default is "nnUNet_prepare_bundle".
    json_file_name : str, optional
        The name of the JSON configuration file. Default is "bundle_config.json".

    Methods
    -------
    handle_event(event_type: str, fl_ctx: FLContext)
        Handles events during the federated learning run. Clears the bundle configuration
        at the start of the run and writes the configuration to a JSON file at the end of the run.
    """
    def __init__(self, results_dir="nnUNet_prepare_bundle", json_file_name="bundle_config.json"):

        super(nnUNetPrepareBundleJsonGenerator, self).__init__()

        self._results_dir = results_dir
        self._bundle_config = {}
        self._json_file_name = json_file_name

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._bundle_config.clear()
        elif event_type == EventType.END_RUN:
            datasets = fl_ctx.get_prop("bundle_config", None)
            run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_job_id())
            cross_val_res_dir = os.path.join(run_dir, self._results_dir)
            if not os.path.exists(cross_val_res_dir):
                os.makedirs(cross_val_res_dir)

            res_file_path = os.path.join(cross_val_res_dir, self._json_file_name)
            with open(res_file_path, "w") as f:
                json.dump(datasets, f)