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


class CSVSaver:
    """
    save the data in a dictionary format cache, and write to a CSV file finally.
    Typically, the data can be classification predictions.
    """

    def __init__(self, output_dir='./', filename='predictions', overwrite=True):
        """
        Args:
            output_dir (str): output CSV file directory.
            filename (str): name of the saved CSV file name.
            overwrite (bool): whether to overwriting existing CSV file content. If we are not overwriting,
                then we check if the results have been previously saved, and load them to the prediction_dict.

        """
        self.output_dir = output_dir
        self._cache_dict = {}
        self._filepath = os.path.join(output_dir, filename + '.csv')
        self.overwrite = overwrite

    def finalize(self):
        """
        Writes the cahced dict to a csv

        """
        if not self.overwrite and os.path.exists(self._filepath):
            with open(self._filepath, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self._cache_dict[row[0]] = np.array(row[1:]).astype(np.float32)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(self._filepath, 'w') as f:
            for k, v in sorted(self._cache_dict.items()):
                f.write(k)
                for result in v.flatten():
                    f.write("," + str(result))
                f.write("\n")

    def save(self, data, meta_data):
        """Save data into Nifti format file. The metadata should have the following key:
            - ``'filename_or_obj'`` -- save the data corresponding to file name or object.

        args:
            data (Tensor or ndarray): target data content that save into cache.
            meta_data (dict): the meta data information corresponding to the data.

        """
        save_key = meta_data['filename_or_obj']
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        self._cache_dict[save_key] = data.astype(np.float32)

    def save_batch(self, batch_data, meta_data):
        """Save a batch of data into the cache dictionary.

        args:
            batch_data (Tensor or ndarray): target batch data content that save into cache.
            meta_data (dict): every key-value in the meta_data is corresponding to 1 batch of data.

        """
        for i, data in enumerate(batch_data):  # save a batch of files
            self.save(data, {k: meta_data[k][i] for k in meta_data})
