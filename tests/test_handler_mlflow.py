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

import glob
import os
import tempfile
import unittest
from pathlib import Path

from ignite.engine import Engine, Events

from monai.handlers import MLFlowHandler


class TestHandlerMLFlow(unittest.TestCase):
    def test_metrics_track(self):
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
                iteration_log=False, epoch_log=True, tracking_uri=Path(test_path).as_uri(), state_attributes=["test"]
            )
            handler.attach(engine)
            engine.run(range(3), max_epochs=2)
            handler.close()
            # check logging output
            self.assertTrue(len(glob.glob(test_path)) > 0)


if __name__ == "__main__":
    unittest.main()
