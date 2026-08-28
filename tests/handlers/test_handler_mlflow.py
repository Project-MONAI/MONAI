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
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
from ignite.engine import Engine, Events
from parameterized import parameterized

from monai.apps import download_and_extract
from monai.bundle import ConfigWorkflow, download
from monai.handlers import MLFlowHandler
from monai.utils import path_to_sqlite_uri, path_to_uri
from tests.test_utils import skip_if_downloading_fails, skip_if_quick


def get_event_filter(e):
    def event_filter(_, event):
        if event in e:
            return True
        return False

    return event_filter


def dummy_train(tracking_folder, tempdir):
    # set up engine
    def _train_func(engine, batch):
        return [batch + 1.0]

    engine = Engine(_train_func)

    # set up testing handler
    test_path = os.path.join(tempdir, tracking_folder)
    handler = MLFlowHandler(
        iteration_log=False,
        epoch_log=True,
        tracking_uri=path_to_sqlite_uri(test_path),
        state_attributes=["test"],
        close_on_complete=True,
    )
    handler.attach(engine)
    engine.run(range(3), max_epochs=2)
    return test_path


class TestHandlerMLFlow(unittest.TestCase):
    def test_multi_run(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # set up the train function for engine
            def _train_func(engine, batch):
                return [batch + 1.0]

            # create and run an engine several times to get several runs
            create_engine_times = 3
            for _ in range(create_engine_times):
                engine = Engine(_train_func)

                @engine.on(Events.EPOCH_COMPLETED)
                def _update_metric(engine):
                    current_metric = engine.state.metrics.get("acc", 0.1)
                    engine.state.metrics["acc"] = current_metric + 0.1
                    engine.state.test = current_metric

                # set up testing handler
                test_path = os.path.join(tempdir, "mlflow_test")
                handler = MLFlowHandler(
                    iteration_log=False,
                    epoch_log=True,
                    tracking_uri=path_to_sqlite_uri(test_path),
                    state_attributes=["test"],
                    close_on_complete=True,
                )
                handler.attach(engine)
                engine.run(range(3), max_epochs=2)
                run_cnt = len(handler.client.search_runs(handler.experiment.experiment_id))
                handler.close()
            # the run count should equal to the times of creating engine
            self.assertEqual(create_engine_times, run_cnt)

    def test_default_tracking_uri_is_sqlite(self):
        """Verify the handler defaults to a local SQLite backend, not the file store, without a tracking URI."""
        with tempfile.TemporaryDirectory() as tempdir:
            cwd = os.getcwd()
            os.chdir(tempdir)
            handler = None
            try:
                handler = MLFlowHandler(iteration_log=False, epoch_log=False)
                self.assertTrue(handler.client.tracking_uri.startswith("sqlite:///"))
                self.assertTrue(handler.client.tracking_uri.endswith("mlruns.db"))
                # artifacts should still default to a `./mlruns`-style directory
                self.assertIsNotNone(handler.artifact_location)
                self.assertTrue(handler.artifact_location.endswith("mlruns"))
            finally:
                if handler is not None:
                    handler.close()  # release the SQLite handle so Windows can delete the db
                os.chdir(cwd)

    def test_remote_tracking_uri_leaves_artifact_location_unset(self):
        """Verify a remote tracking URI gets no local artifact location injected."""
        handler = MLFlowHandler(iteration_log=False, epoch_log=False, tracking_uri="http://localhost:5000")
        self.assertEqual(handler.client.tracking_uri, "http://localhost:5000")
        self.assertIsNone(handler.artifact_location)

    def test_file_store_tracking_uri_is_rejected(self):
        """Verify local paths and file:// URIs are rejected with an actionable error."""
        for uri in ("/tmp/mlruns", path_to_uri(os.path.join("some", "dir"))):
            with self.assertRaises(ValueError):
                MLFlowHandler(iteration_log=False, epoch_log=False, tracking_uri=uri)

    def test_explicit_sqlite_tracking_uri_colocates_artifacts(self):
        """Verify an explicit SQLite tracking URI co-locates artifacts next to the database."""
        with tempfile.TemporaryDirectory() as tempdir:
            uri = path_to_sqlite_uri(os.path.join(tempdir, "sub", "mlruns.db"))
            handler = MLFlowHandler(iteration_log=False, epoch_log=False, tracking_uri=uri)
            try:
                self.assertEqual(handler.client.tracking_uri, uri)
                self.assertIsNotNone(handler.artifact_location)
                self.assertTrue(handler.artifact_location.endswith("mlruns"))
            finally:
                handler.close()  # release the SQLite handle so Windows can delete the db

    def test_env_var_sqlite_tracking_uri_colocates_artifacts(self):
        """Verify a SQLite ``MLFLOW_TRACKING_URI`` env var co-locates artifacts next to the db."""
        with tempfile.TemporaryDirectory() as tempdir:
            uri = path_to_sqlite_uri(os.path.join(tempdir, "sub", "mlruns.db"))
            handler = None
            with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": uri}):
                try:
                    handler = MLFlowHandler(iteration_log=False, epoch_log=False)
                    self.assertTrue(handler.client.tracking_uri.endswith("mlruns.db"))
                    self.assertIsNotNone(handler.artifact_location)
                    self.assertTrue(handler.artifact_location.endswith("mlruns"))
                    # co-located with the db file (the `sub` dir), not a cwd-relative `./mlruns`
                    self.assertIn("sub", handler.artifact_location)
                finally:
                    if handler is not None:
                        handler.close()  # release the SQLite handle so Windows can delete the db

    def test_env_var_tracking_uri_takes_priority_over_argument(self):
        """Verify ``MLFLOW_TRACKING_URI`` overrides an explicit ``tracking_uri`` argument."""
        with tempfile.TemporaryDirectory() as tempdir:
            env_uri = path_to_sqlite_uri(os.path.join(tempdir, "env.db"))
            arg_uri = path_to_sqlite_uri(os.path.join(tempdir, "arg.db"))
            handler = None
            with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": env_uri}):
                try:
                    handler = MLFlowHandler(iteration_log=False, epoch_log=False, tracking_uri=arg_uri)
                    self.assertTrue(handler.client.tracking_uri.endswith("env.db"))
                finally:
                    if handler is not None:
                        handler.close()  # release the SQLite handle so Windows can delete the db

    def test_explicit_artifact_location_is_used(self):
        """Verify an explicit artifact location is preserved with the default SQLite backend."""
        with tempfile.TemporaryDirectory() as tempdir:
            cwd = os.getcwd()
            os.chdir(tempdir)
            handler = None
            try:
                art = path_to_uri(os.path.join(tempdir, "artifacts"))
                handler = MLFlowHandler(iteration_log=False, epoch_log=False, artifact_location=art)
                self.assertEqual(handler.artifact_location, art)
            finally:
                if handler is not None:
                    handler.close()  # release the SQLite handle so Windows can delete the db
                os.chdir(cwd)

    def test_default_sqlite_run_flow(self):
        """Verify a basic run flow works end-to-end with the default SQLite backend."""
        with tempfile.TemporaryDirectory() as tempdir:
            cwd = os.getcwd()
            os.chdir(tempdir)
            try:

                def _train_func(engine, batch):
                    return [batch + 1.0]

                engine = Engine(_train_func)

                @engine.on(Events.EPOCH_COMPLETED)
                def _update_metric(engine):
                    current_metric = engine.state.metrics.get("acc", 0.1)
                    engine.state.metrics["acc"] = current_metric + 0.1

                # close_on_complete=False so cur_run stays available after the run for the metric
                # check below; the run is closed explicitly afterwards.
                handler = MLFlowHandler(iteration_log=False, epoch_log=True, close_on_complete=False)
                handler.attach(engine)
                engine.run(range(3), max_epochs=2)
                cur_run = handler.client.get_run(handler.cur_run.info.run_id)
                self.assertTrue("acc" in cur_run.data.metrics.keys())
                handler.close()
                # the default backend should have created a SQLite database file in the cwd
                self.assertTrue(os.path.exists(os.path.join(tempdir, "mlruns.db")))
            finally:
                os.chdir(cwd)

    def test_metrics_track(self):
        experiment_param = {"backbone": "efficientnet_b0"}
        with tempfile.TemporaryDirectory() as tempdir:
            # set up engine
            def _train_func(engine, batch):
                return [batch + 1.0]

            engine = Engine(_train_func)

            # set up dummy metric
            @engine.on(Events.EPOCH_COMPLETED)
            def _update_metric(engine):
                current_metric = engine.state.metrics.get("acc", 0.1)
                engine.state.metrics["acc"] = current_metric + 0.1
                # log nested metrics
                engine.state.metrics["acc_per_label"] = {
                    "label_0": current_metric + 0.1,
                    "label_1": current_metric + 0.2,
                }
                engine.state.test = current_metric

            # set up testing handler
            test_path = os.path.join(tempdir, "mlflow_test")
            artifact_path = os.path.join(tempdir, "artifacts")
            os.makedirs(artifact_path, exist_ok=True)
            dummy_numpy = np.zeros((64, 64, 3))
            dummy_path = os.path.join(artifact_path, "tmp.npy")
            np.save(dummy_path, dummy_numpy)
            handler = MLFlowHandler(
                iteration_log=False,
                epoch_log=True,
                tracking_uri=path_to_sqlite_uri(test_path),
                state_attributes=["test"],
                experiment_param=experiment_param,
                artifacts=[artifact_path],
                close_on_complete=False,
            )
            handler.attach(engine)
            engine.run(range(3), max_epochs=2)
            cur_run = handler.client.get_run(handler.cur_run.info.run_id)
            self.assertTrue("label_0" in cur_run.data.metrics.keys())
            handler.close()
            # check logging output
            self.assertTrue(len(glob.glob(test_path)) > 0)

    @parameterized.expand([[True], [get_event_filter([1, 2])]])
    def test_metrics_track_mock(self, epoch_log):
        experiment_param = {"backbone": "efficientnet_b0"}
        with tempfile.TemporaryDirectory() as tempdir:
            # set up engine
            def _train_func(engine, batch):
                return [batch + 1.0]

            engine = Engine(_train_func)

            # set up dummy metric
            @engine.on(Events.EPOCH_COMPLETED)
            def _update_metric(engine):
                current_metric = engine.state.metrics.get("acc", 0.1)
                engine.state.metrics["acc"] = current_metric + 0.1
                engine.state.test = current_metric

            # set up testing handler
            test_path = os.path.join(tempdir, "mlflow_test")
            handler = MLFlowHandler(
                iteration_log=False,
                epoch_log=epoch_log,
                tracking_uri=path_to_sqlite_uri(test_path),
                state_attributes=["test"],
                experiment_param=experiment_param,
                close_on_complete=True,
            )
            handler._default_epoch_log = MagicMock()
            handler.attach(engine)

            max_epochs = 4
            engine.run(range(3), max_epochs=max_epochs)
            handler.close()
            # check logging output
            if epoch_log is True:
                self.assertEqual(handler._default_epoch_log.call_count, max_epochs)
            else:
                self.assertEqual(handler._default_epoch_log.call_count, 2)  # 2 = len([1, 2]) from event_filter

    @parameterized.expand([[True], [get_event_filter([1, 3])]])
    def test_metrics_track_iters_mock(self, iteration_log):
        experiment_param = {"backbone": "efficientnet_b0"}
        with tempfile.TemporaryDirectory() as tempdir:
            # set up engine
            def _train_func(engine, batch):
                return [batch + 1.0]

            engine = Engine(_train_func)

            # set up dummy metric
            @engine.on(Events.EPOCH_COMPLETED)
            def _update_metric(engine):
                current_metric = engine.state.metrics.get("acc", 0.1)
                engine.state.metrics["acc"] = current_metric + 0.1
                engine.state.test = current_metric

            # set up testing handler
            test_path = os.path.join(tempdir, "mlflow_test")
            handler = MLFlowHandler(
                iteration_log=iteration_log,
                epoch_log=False,
                tracking_uri=path_to_sqlite_uri(test_path),
                state_attributes=["test"],
                experiment_param=experiment_param,
                close_on_complete=True,
            )
            handler._default_iteration_log = MagicMock()
            handler.attach(engine)

            num_iters = 3
            max_epochs = 2
            engine.run(range(num_iters), max_epochs=max_epochs)
            handler.close()
            # check logging output
            if iteration_log is True:
                self.assertEqual(handler._default_iteration_log.call_count, num_iters * max_epochs)
            else:
                self.assertEqual(handler._default_iteration_log.call_count, 2)  # 2 = len([1, 3]) from event_filter

    def test_multi_thread(self):
        test_uri_list = ["monai_mlflow_test1", "monai_mlflow_test2"]
        with tempfile.TemporaryDirectory() as tempdir:
            with ThreadPoolExecutor(2, "Training") as executor:
                futures = {}
                for t in test_uri_list:
                    futures[t] = executor.submit(dummy_train, t, tempdir)

                for _, future in futures.items():
                    res = future.result()
                    self.assertTrue(len(glob.glob(res)) > 0)

    @skip_if_quick
    def test_dataset_tracking(self):
        test_bundle_name = "endoscopic_tool_segmentation"
        with tempfile.TemporaryDirectory() as tempdir:
            resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/endoscopic_tool_dataset.zip"
            md5 = "f82da47259c0a617202fb54624798a55"
            compressed_file = os.path.join(tempdir, "endoscopic_tool_segmentation.zip")
            data_dir = os.path.join(tempdir, "endoscopic_tool_dataset")
            with skip_if_downloading_fails():
                if not os.path.exists(data_dir):
                    download_and_extract(resource, compressed_file, tempdir, md5)

                download(test_bundle_name, bundle_dir=tempdir)

                bundle_root = os.path.join(tempdir, test_bundle_name)
                config_file = os.path.join(bundle_root, "configs/inference.json")
                meta_file = os.path.join(bundle_root, "configs/metadata.json")
                logging_file = os.path.join(bundle_root, "configs/logging.conf")
                workflow = ConfigWorkflow(
                    workflow_type="infer",
                    config_file=config_file,
                    meta_file=meta_file,
                    logging_file=logging_file,
                    init_id="initialize",
                    run_id="run",
                    final_id="finalize",
                )

                tracking_path = os.path.join(tempdir, "mlflow_dataset.db")
                workflow.bundle_root = bundle_root
                workflow.dataset_dir = data_dir
                workflow.initialize()
                infer_dataset = workflow.dataset
                mlflow_handler = MLFlowHandler(
                    iteration_log=False,
                    epoch_log=False,
                    dataset_dict={"test": infer_dataset},
                    tracking_uri=path_to_sqlite_uri(tracking_path),
                )
                mlflow_handler.attach(workflow.evaluator)
                workflow.run()
                workflow.finalize()

                cur_run = mlflow_handler.client.get_run(mlflow_handler.cur_run.info.run_id)
                logged_nontrain_set = [x for x in cur_run.inputs.dataset_inputs if x.dataset.name.startswith("test")]
                self.assertEqual(len(logged_nontrain_set), 1)
                mlflow_handler.close()


if __name__ == "__main__":
    unittest.main()
