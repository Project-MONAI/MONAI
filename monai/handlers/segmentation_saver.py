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

from ignite.engine import Events
import logging
from monai.data.nifti_saver import NiftiSaver


class SegmentationSaver:
    """
    Event handler triggered on completing every iteration to save the segmentation predictions as nifti files.
    """

    def __init__(self, output_dir='./', output_postfix='seg', output_ext='.nii.gz', dtype=None,
                 batch_transform=lambda x: x, output_transform=lambda x: x, name=None):
        """
        Args:
            output_dir (str): output image directory.
            output_postfix (str): a string appended to all output file names.
            output_ext (str): output file extension name.
            dtype (np.dtype, optional): convert the image data to save to this data type.
                If None, keep the original type of data.
            batch_transform (Callable): a callable that is used to transform the
                ignite.engine.batch into expected format to extract the meta_data dictionary.
            output_transform (Callable): a callable that is used to transform the
                ignite.engine.output into the form expected nifti image data.
                The first dimension of this transform's output will be treated as the
                batch dimension. Each item in the batch will be saved individually.
            name (str): identifier of logging.logger to use, defaulting to `engine.logger`.

        """
        self.saver = NiftiSaver(output_dir, output_postfix, output_ext, dtype)
        self.batch_transform = batch_transform
        self.output_transform = output_transform

        self.logger = None if name is None else logging.getLogger(name)

    def attach(self, engine):
        if self.logger is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine):
        """
        This method assumes self.batch_transform will extract metadata from the input batch.
        output file datatype is determined from ``engine.state.output.dtype``.

        """
        meta_data = self.batch_transform(engine.state.batch)
        engine_output = self.output_transform(engine.state.output)
        self.saver.save_batch(engine_output, meta_data)
        self.logger.info('saved all the model outputs as nifti files.')
