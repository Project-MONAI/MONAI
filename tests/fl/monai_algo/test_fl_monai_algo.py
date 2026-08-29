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

import glob
import json
import logging
import os
import shutil
import tempfile
import unittest
import warnings
from copy import deepcopy
from os.path import join as pathjoin
from pathlib import Path

from parameterized import parameterized

from monai.bundle import ConfigParser, ConfigWorkflow
from monai.bundle.utils import DEFAULT_HANDLERS_ID
from monai.fl.client.monai_algo import MonaiAlgo, MonaiAlgoStats
from monai.fl.utils.constants import ExtraItems
from monai.fl.utils.exchange_object import ExchangeObject
from monai.utils import path_to_sqlite_uri
from tests.test_utils import SkipIfNoModule

_root_dir = Path(__file__).resolve().parents[2]
_data_dir = os.path.join(_root_dir, "testing_data")
_logging_file = pathjoin(_data_dir, "logging.conf")

TEST_TRAIN_1 = [
    {
        "bundle_root": _data_dir,
        "train_workflow": ConfigWorkflow(
            config_file=os.path.join(_data_dir, "config_fl_train.json"),
            workflow_type="train",
            logging_file=_logging_file,
        ),
        "config_evaluate_filename": None,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_TRAIN_2 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": os.path.join(_data_dir, "config_fl_train.json"),
        "config_evaluate_filename": None,
        "config_filters_filename": None,
    }
]
TEST_TRAIN_3 = [
    {
        "bundle_root": _data_dir,
        "train_workflow": ConfigWorkflow(
            config_file=os.path.join(_data_dir, "config_fl_train.json"),
            workflow_type="train",
            logging_file=_logging_file,
        ),
        "config_evaluate_filename": None,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]

TEST_TRAIN_4 = [
    {
        "bundle_root": _data_dir,
        "train_workflow": ConfigWorkflow(
            config_file=os.path.join(_data_dir, "config_fl_train.json"),
            workflow_type="train",
            logging_file=_logging_file,
            tracking={
                "handlers_id": DEFAULT_HANDLERS_ID,
                "configs": {
                    "save_execute_config": f"{_data_dir}/config_executed.json",
                    "trainer": {
                        "_target_": "MLFlowHandler",
                        "tracking_uri": path_to_sqlite_uri(os.path.join(_data_dir, "mlflow_override.db")),
                        "output_transform": "$monai.handlers.from_engine(['loss'], first=True)",
                        "close_on_complete": True,
                    },
                },
            },
        ),
        "config_evaluate_filename": None,
        "config_filters_filename": None,
    }
]

TEST_EVALUATE_1 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": None,
        "eval_workflow": ConfigWorkflow(
            config_file=[
                os.path.join(_data_dir, "config_fl_train.json"),
                os.path.join(_data_dir, "config_fl_evaluate.json"),
            ],
            workflow_type="train",
            logging_file=_logging_file,
            tracking="mlflow",
            tracking_uri=path_to_sqlite_uri(os.path.join(_data_dir, "mlflow_1.db")),
            experiment_name="monai_eval1",
        ),
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_EVALUATE_2 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": None,
        "config_evaluate_filename": [
            os.path.join(_data_dir, "config_fl_train.json"),
            os.path.join(_data_dir, "config_fl_evaluate.json"),
        ],
        "eval_kwargs": {
            "tracking": "mlflow",
            "tracking_uri": path_to_sqlite_uri(os.path.join(_data_dir, "mlflow_2.db")),
            "experiment_name": "monai_eval2",
        },
        "eval_workflow_name": "training",
        "config_filters_filename": None,
    }
]
TEST_EVALUATE_3 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": None,
        "eval_workflow": ConfigWorkflow(
            config_file=[
                os.path.join(_data_dir, "config_fl_train.json"),
                os.path.join(_data_dir, "config_fl_evaluate.json"),
            ],
            workflow_type="train",
            logging_file=_logging_file,
        ),
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]

TEST_GET_WEIGHTS_1 = [
    {
        "bundle_root": _data_dir,
        "train_workflow": ConfigWorkflow(
            config_file=os.path.join(_data_dir, "config_fl_train.json"),
            workflow_type="train",
            logging_file=_logging_file,
        ),
        "config_evaluate_filename": None,
        "send_weight_diff": False,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_GET_WEIGHTS_2 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": os.path.join(_data_dir, "config_fl_train.json"),
        "config_evaluate_filename": None,
        "send_weight_diff": True,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_GET_WEIGHTS_3 = [
    {
        "bundle_root": _data_dir,
        "train_workflow": ConfigWorkflow(
            config_file=os.path.join(_data_dir, "config_fl_train.json"),
            workflow_type="train",
            logging_file=_logging_file,
        ),
        "config_evaluate_filename": None,
        "send_weight_diff": True,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]


def _dispose_sqlite_engines():
    """Dispose MLflow's open SQLAlchemy SQLite engines so the test ``.db`` files can be removed.

    MLflow keeps a SQLite connection open for the lifetime of its client; on Windows that
    locks the database file and breaks cleanup. ``MLFlowHandler.close()`` releases it, but a
    workflow may finish without closing every handler, so dispose defensively here before
    deleting the files. Scoped to the test's ``mlflow*.db`` backends so unrelated (e.g.
    in-memory) sqlite engines elsewhere in the process are left untouched.
    """
    import gc

    try:
        from sqlalchemy.engine import Engine
    except ImportError:
        return
    gc.collect()
    for obj in gc.get_objects():
        # gc.get_objects() can include dead weakref proxies, whose isinstance() raises
        # ReferenceError, so guard the whole inspection (ReferenceError is an Exception).
        try:
            if not isinstance(obj, Engine):
                continue
            url = obj.url
            db = url.database if url.get_backend_name() == "sqlite" else None
            # the test backends are all files named ``mlflow*.db``; match those only so
            # unrelated (e.g. in-memory) sqlite engines in the process are left untouched.
            if db and os.path.basename(db).startswith("mlflow"):
                obj.dispose()
        except Exception:
            pass


@SkipIfNoModule("ignite")
@SkipIfNoModule("mlflow")
class TestFLMonaiAlgo(unittest.TestCase):
    @parameterized.expand([TEST_TRAIN_1, TEST_TRAIN_2, TEST_TRAIN_3, TEST_TRAIN_4])
    def test_train(self, input_params):
        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})
        algo.abort()

        # initialize model
        parser = ConfigParser(config=deepcopy(algo.train_workflow.parser.get()))
        parser.parse()
        network = parser.get_parsed_content("network")

        data = ExchangeObject(weights=network.state_dict())

        # test train
        algo.train(data=data, extra={})
        algo.finalize()

        # test experiment management
        if "save_execute_config" in algo.train_workflow.parser:
            _dispose_sqlite_engines()  # release SQLite handles so the db file can be removed on Windows
            self.assertTrue(os.path.exists(f"{_data_dir}/mlflow_override.db"))
            os.remove(f"{_data_dir}/mlflow_override.db")
            if os.path.isdir(f"{_data_dir}/mlruns"):
                shutil.rmtree(f"{_data_dir}/mlruns")
            self.assertTrue(os.path.exists(f"{_data_dir}/config_executed.json"))
            os.remove(f"{_data_dir}/config_executed.json")

    @parameterized.expand([TEST_EVALUATE_1, TEST_EVALUATE_2, TEST_EVALUATE_3])
    def test_evaluate(self, input_params):
        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})

        # initialize model
        parser = ConfigParser(config=deepcopy(algo.eval_workflow.parser.get()))
        parser.parse()
        network = parser.get_parsed_content("network")

        data = ExchangeObject(weights=network.state_dict())

        # test evaluate
        algo.evaluate(data=data, extra={})

        # test experiment management
        if "save_execute_config" in algo.eval_workflow.parser:
            _dispose_sqlite_engines()  # release SQLite handles so the db files can be removed on Windows
            self.assertGreater(len(list(glob.glob(f"{_data_dir}/mlflow_*"))), 0)
            for f in list(glob.glob(f"{_data_dir}/mlflow_*")):
                shutil.rmtree(f) if os.path.isdir(f) else os.remove(f)
            if os.path.isdir(f"{_data_dir}/mlruns"):
                shutil.rmtree(f"{_data_dir}/mlruns")
            self.assertGreater(len(list(glob.glob(f"{_data_dir}/eval/config_*"))), 0)
            for f in list(glob.glob(f"{_data_dir}/eval/config_*")):
                os.remove(f)

    @parameterized.expand([TEST_GET_WEIGHTS_1, TEST_GET_WEIGHTS_2, TEST_GET_WEIGHTS_3])
    def test_get_weights(self, input_params):
        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})

        # test train
        if input_params["send_weight_diff"]:  # should not work as test doesn't receive a global model
            with self.assertRaises(ValueError):
                weights = algo.get_weights(extra={})
        else:
            weights = algo.get_weights(extra={})
            self.assertIsInstance(weights, ExchangeObject)


@SkipIfNoModule("ignite")
class TestFLMonaiAlgoWarnsOnProvisionedConfig(unittest.TestCase):
    """Regression tests for GHSA-x6pr-233j-x5cw: `MonaiAlgo`/`MonaiAlgoStats` execute a bundle whose
    whole app directory -- configs included -- is provisioned by the FL system, and the aggregation
    server dispatches the tasks that run it with no per-round human interaction. `MonaiAlgo` builds
    its `ConfigWorkflow` directly rather than through `create_workflow()`, so the warning added for
    GHSA-873f-pvrv-4x83 never fired on this path.

    Executing the config is still not blocked -- MONAI has no way to establish whether a bundle is
    trustworthy, so a flag would only teach operators to set it once and forget it -- but a
    `UserWarning` is now raised, and the one sink with no functional role in FL, the bundle's own
    "configs/logging.conf", is no longer applied unless the FL system asks for it via
    `ExtraItems.LOGGING_FILE` (GHSA-wvpx-5qmp-46g3)."""

    def setUp(self):
        # `fileConfig` reconfigures logging process-wide. Snapshot the root logger and restore it
        # afterwards so these tests cannot leak a handler into the rest of the suite.
        root = logging.getLogger()
        level, handlers, filters = root.level, root.handlers[:], root.filters[:]
        disabled = logging.root.manager.disable

        def _restore():
            # Detach whatever is on the root logger now, closing anything `fileConfig` installed so
            # it does not linger in logging's handler registry, then put the snapshot back. Under
            # `tests/runner.py` the root logger starts with no handlers, so there is nothing for
            # `fileConfig` to have closed on the way in.
            for handler in root.handlers[:]:
                root.removeHandler(handler)
                if handler not in handlers:
                    handler.close()
            root.setLevel(level)
            root.filters[:] = filters
            for handler in handlers:
                root.addHandler(handler)
            logging.disable(disabled)

        self.addCleanup(_restore)

    def _stage_malicious_app(self, tempdir: str) -> tuple[str, str, str]:
        """Write an FL app whose config and logging.conf each drop a distinct marker file."""
        app_root = os.path.join(tempdir, "app")
        os.makedirs(os.path.join(app_root, "configs"))
        config_marker = os.path.join(tempdir, "CONFIG_PWNED")
        logging_marker = os.path.join(tempdir, "LOGGING_PWNED")
        # write the markers via `pathlib` instead of shelling out through `os.system` -- `!r` yields
        # a Python-source-safe literal (handling spaces and Windows backslashes alike) with no shell
        # involved to reintroduce quoting/splitting issues.
        payload = f"$__import__('pathlib').Path({config_marker!r}).write_text('pwned')"
        with open(os.path.join(app_root, "configs", "train.json"), "w") as f:
            json.dump({"initialize": [payload]}, f)
        # `fileConfig` eval()s the `class=` field, so the tuple subscript runs the payload and still
        # yields a usable handler class.
        with open(os.path.join(app_root, "configs", "logging.conf"), "w") as f:
            f.write(
                "[loggers]\nkeys=root\n[handlers]\nkeys=h\n[formatters]\nkeys=f\n"
                "[logger_root]\nlevel=NOTSET\nhandlers=h\n"
                "[handler_h]\n"
                f"class=(__import__('pathlib').Path({logging_marker!r}).write_text('pwned'), "
                "__import__('logging').StreamHandler)[1]\nargs=()\nformatter=f\n"
                "[formatter_f]\nformat=%(message)s\n"
            )
        return app_root, config_marker, logging_marker

    @staticmethod
    def _algo(algo_class):
        # `bundle_root=""` so the whole path comes from the server-supplied `APP_ROOT`, exactly as in
        # the advisory's PoC. `MonaiAlgo` additionally defaults to building an evaluate workflow.
        kwargs = {"bundle_root": "", "config_train_filename": "configs/train.json"}
        if algo_class is MonaiAlgo:
            kwargs["config_evaluate_filename"] = None
        return algo_class(**kwargs)

    @parameterized.expand([[MonaiAlgoStats], [MonaiAlgo]])
    def test_warns_and_executes_provisioned_config(self, algo_class):
        with tempfile.TemporaryDirectory() as tempdir:
            app_root, config_marker, logging_marker = self._stage_malicious_app(tempdir)
            algo = self._algo(algo_class)
            with self.assertWarnsRegex(UserWarning, r"GHSA-x6pr-233j-x5cw"):
                # the staged config defines only `initialize`, so resolving the `bundle_root`
                # property fails *after* the payload has already run -- as in the advisory's own
                # PoC, where the failure happens after code execution.
                with self.assertRaises(KeyError):
                    algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl", ExtraItems.APP_ROOT: app_root})
            # executing the config is deliberately still not blocked
            self.assertTrue(os.path.exists(config_marker))
            # ... but the server's logging.conf is no longer handed to `fileConfig`
            self.assertFalse(os.path.exists(logging_marker))

    @parameterized.expand([[MonaiAlgoStats], [MonaiAlgo]])
    def test_explicit_none_logging_file_does_not_apply_provisioned_conf(self, algo_class):
        """`None` was the pre-fix default, so an FL system may well pass the key explicitly with that
        value. `ConfigWorkflow` reads `None` as "fall back to the bundle's own configs/logging.conf",
        which would hand the server's INI straight to `fileConfig`; it has to mean disabled here."""
        with tempfile.TemporaryDirectory() as tempdir:
            app_root, _, logging_marker = self._stage_malicious_app(tempdir)
            algo = self._algo(algo_class)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with self.assertRaises(KeyError):
                    algo.initialize(
                        extra={
                            ExtraItems.CLIENT_NAME: "test_fl",
                            ExtraItems.APP_ROOT: app_root,
                            ExtraItems.LOGGING_FILE: None,
                        }
                    )
            self.assertFalse(any("GHSA-wvpx-5qmp-46g3" in str(w.message) for w in caught))
            self.assertFalse(os.path.exists(logging_marker))

    @parameterized.expand([[MonaiAlgoStats], [MonaiAlgo]])
    def test_logging_file_opt_in_applies_provisioned_conf(self, algo_class):
        with tempfile.TemporaryDirectory() as tempdir:
            app_root, _, logging_marker = self._stage_malicious_app(tempdir)
            algo = self._algo(algo_class)
            with self.assertWarnsRegex(UserWarning, r"GHSA-wvpx-5qmp-46g3"):
                with self.assertRaises(KeyError):
                    algo.initialize(
                        extra={
                            ExtraItems.CLIENT_NAME: "test_fl",
                            ExtraItems.APP_ROOT: app_root,
                            ExtraItems.LOGGING_FILE: os.path.join(app_root, "configs", "logging.conf"),
                        }
                    )
            self.assertTrue(os.path.exists(logging_marker))

    def test_no_logging_warning_when_logging_disabled(self):
        """The `fileConfig` warning must not fire when nothing is actually executed."""
        with tempfile.TemporaryDirectory() as tempdir:
            app_root, _, _ = self._stage_malicious_app(tempdir)
            algo = self._algo(MonaiAlgoStats)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with self.assertRaises(KeyError):
                    algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl", ExtraItems.APP_ROOT: app_root})
            self.assertFalse(any("GHSA-wvpx-5qmp-46g3" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
