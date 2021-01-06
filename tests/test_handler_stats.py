# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
import tempfile
import unittest
from io import StringIO

import torch
from ignite.engine import Engine, Events

from monai.handlers import StatsHandler


class TestHandlerStats(unittest.TestCase):
    def test_metrics_print(self):
        log_stream = StringIO()
        logging.basicConfig(stream=log_stream, level=logging.INFO)
        key_to_handler = "test_logging"
        key_to_print = "testing_metric"

        # set up engine
        def _train_func(engine, batch):
            return torch.tensor(0.0)

        engine = Engine(_train_func)

        # set up dummy metric
        @engine.on(Events.EPOCH_COMPLETED)
        def _update_metric(engine):
            current_metric = engine.state.metrics.get(key_to_print, 0.1)
            engine.state.metrics[key_to_print] = current_metric + 0.1

        # set up testing handler
        stats_handler = StatsHandler(name=key_to_handler)
        stats_handler.attach(engine)

        engine.run(range(3), max_epochs=2)

        # check logging output
        output_str = log_stream.getvalue()
        grep = re.compile(f".*{key_to_handler}.*")
        has_key_word = re.compile(f".*{key_to_print}.*")
        for idx, line in enumerate(output_str.split("\n")):
            if grep.match(line):
                if idx in [5, 10]:
                    self.assertTrue(has_key_word.match(line))

    def test_loss_print(self):
        log_stream = StringIO()
        logging.basicConfig(stream=log_stream, level=logging.INFO)
        key_to_handler = "test_logging"
        key_to_print = "myLoss"

        # set up engine
        def _train_func(engine, batch):
            return torch.tensor(0.0)

        engine = Engine(_train_func)

        # set up testing handler
        stats_handler = StatsHandler(name=key_to_handler, tag_name=key_to_print)
        stats_handler.attach(engine)

        engine.run(range(3), max_epochs=2)

        # check logging output
        output_str = log_stream.getvalue()
        grep = re.compile(f".*{key_to_handler}.*")
        has_key_word = re.compile(f".*{key_to_print}.*")
        for idx, line in enumerate(output_str.split("\n")):
            if grep.match(line):
                if idx in [1, 2, 3, 6, 7, 8]:
                    self.assertTrue(has_key_word.match(line))

    def test_loss_dict(self):
        log_stream = StringIO()
        logging.basicConfig(stream=log_stream, level=logging.INFO)
        key_to_handler = "test_logging"
        key_to_print = "myLoss1"

        # set up engine
        def _train_func(engine, batch):
            return torch.tensor(0.0)

        engine = Engine(_train_func)

        # set up testing handler
        stats_handler = StatsHandler(name=key_to_handler, output_transform=lambda x: {key_to_print: x})
        stats_handler.attach(engine)

        engine.run(range(3), max_epochs=2)

        # check logging output
        output_str = log_stream.getvalue()
        grep = re.compile(f".*{key_to_handler}.*")
        has_key_word = re.compile(f".*{key_to_print}.*")
        for idx, line in enumerate(output_str.split("\n")):
            if grep.match(line):
                if idx in [1, 2, 3, 6, 7, 8]:
                    self.assertTrue(has_key_word.match(line))

    def test_loss_file(self):
        logging.basicConfig(level=logging.INFO)
        key_to_handler = "test_logging"
        key_to_print = "myLoss"

        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_loss_stats.log")
            handler = logging.FileHandler(filename, mode="w")

            # set up engine
            def _train_func(engine, batch):
                return torch.tensor(0.0)

            engine = Engine(_train_func)

            # set up testing handler
            stats_handler = StatsHandler(name=key_to_handler, tag_name=key_to_print, logger_handler=handler)
            stats_handler.attach(engine)

            engine.run(range(3), max_epochs=2)
            handler.stream.close()
            stats_handler.logger.removeHandler(handler)
            with open(filename, "r") as f:
                output_str = f.read()
                grep = re.compile(f".*{key_to_handler}.*")
                has_key_word = re.compile(f".*{key_to_print}.*")
                for idx, line in enumerate(output_str.split("\n")):
                    if grep.match(line):
                        if idx in [1, 2, 3, 6, 7, 8]:
                            self.assertTrue(has_key_word.match(line))

    def test_exception(self):
        logging.basicConfig(level=logging.INFO)

        # set up engine
        def _train_func(engine, batch):
            raise RuntimeError("test exception.")

        engine = Engine(_train_func)

        # set up testing handler
        stats_handler = StatsHandler()
        stats_handler.attach(engine)

        with self.assertRaises(RuntimeError):
            engine.run(range(3), max_epochs=2)


if __name__ == "__main__":
    unittest.main()
