# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import csv
import numpy as np
import torch
from ignite.engine import Events
import logging


class ClassificationSaver:
    """
    Event handler triggered on completing every iteration to save the classification predictions as CSV file.
    """

    def __init__(self, output_dir='./', overwrite=True,
                 batch_transform=lambda x: x, output_transform=lambda x: x, name=None):
        """
        Args:
            output_dir (str): output CSV file directory.
            overwrite (bool): whether to overwriting existing CSV file content. If we are not overwriting,
                then we check if the results have been previously saved, and load them to the prediction_dict.
            batch_transform (Callable): a callable that is used to transform the
                ignite.engine.batch into expected format to extract the meta_data dictionary.
            output_transform (Callable): a callable that is used to transform the
                ignite.engine.output into the form expected model prediction data.
                The first dimension of this transform's output will be treated as the
                batch dimension. Each item in the batch will be saved individually.
            name (str): identifier of logging.logger to use, defaulting to `engine.logger`.

        """
        self.output_dir = output_dir
        self._prediction_dict = {}
        self._preds_filepath = os.path.join(output_dir, 'predictions.csv')
        self.overwrite = overwrite
        self.batch_transform = batch_transform
        self.output_transform = output_transform

        self.logger = None if name is None else logging.getLogger(name)

    def attach(self, engine):
        if self.logger is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        if not engine.has_event_handler(self.finalize, Events.COMPLETED):
            engine.add_event_handler(Events.COMPLETED, self.finalize)

    def finalize(self, _engine=None):
        """
        Writes the prediction dict to a csv

        """
        if not self.overwrite and os.path.exists(self._preds_filepath):
            with open(self._preds_filepath, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self._prediction_dict[row[0]] = np.array(row[1:]).astype(np.float32)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(self._preds_filepath, 'w') as f:
            for k, v in sorted(self._prediction_dict.items()):
                f.write(k)
                for result in v.flatten():
                    f.write("," + str(result))
                f.write("\n")
        self.logger.info('saved classification predictions into: {}'.format(self._preds_filepath))

    def __call__(self, engine):
        """
        This method assumes self.batch_transform will extract Metadata from the input batch.
        Metadata should have the following keys:

            - ``'filename_or_obj'`` -- save the prediction corresponding to file name.

        """
        meta_data = self.batch_transform(engine.state.batch)
        filenames = meta_data['filename_or_obj']

        engine_output = self.output_transform(engine.state.output)
        for batch_id, filename in enumerate(filenames):  # save a batch of files
            output = engine_output[batch_id]
            if isinstance(output, torch.Tensor):
                output = output.detach().cpu().numpy()
            self._prediction_dict[filename] = output.astype(np.float32)
