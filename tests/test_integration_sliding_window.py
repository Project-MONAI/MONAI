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
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch
from ignite.engine import Engine
from torch.utils.data import DataLoader

from monai.data.nifti_reader import NiftiDataset
from monai.data.synthetic import create_test_image_3d
from monai.handlers.segmentation_saver import SegmentationSaver
from monai.networks.nets.unet import UNet
from monai.networks.utils import predict_segmentation
from monai.transforms.transforms import AddChannel
from monai.utils.sliding_window_inference import sliding_window_inference
from tests.utils import make_nifti_image


def run_test(batch_size, img_name, seg_name, output_dir, device=torch.device("cuda:0")):
    ds = NiftiDataset([img_name], [seg_name], transform=AddChannel(), seg_transform=AddChannel(), image_only=False)
    loader = DataLoader(ds, batch_size=1, pin_memory=torch.cuda.is_available())

    net = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(4, 8, 16, 32),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)
    roi_size = (16, 32, 48)
    sw_batch_size = batch_size

    def _sliding_window_processor(_engine, batch):
        net.eval()
        img, seg, meta_data = batch
        with torch.no_grad():
            seg_probs = sliding_window_inference(img.to(device), roi_size, sw_batch_size, net)
            return predict_segmentation(seg_probs)

    infer_engine = Engine(_sliding_window_processor)

    SegmentationSaver(output_dir=output_dir, output_ext='.nii.gz', output_postfix='seg',
                      batch_transform=lambda x: x[2]).attach(infer_engine)

    infer_engine.run(loader)

    basename = os.path.basename(img_name)[:-len('.nii.gz')]
    saved_name = os.path.join(output_dir, basename, '{}_seg.nii.gz'.format(basename))
    return saved_name


class TestIntegrationSlidingWindow(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        im, seg = create_test_image_3d(25, 28, 63, rad_max=10, noise_max=1, num_objs=4, num_seg_classes=1)
        self.img_name = make_nifti_image(im)
        self.seg_name = make_nifti_image(seg)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

    def tearDown(self):
        if os.path.exists(self.img_name):
            os.remove(self.img_name)
        if os.path.exists(self.seg_name):
            os.remove(self.seg_name)

    def test_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = run_test(batch_size=2,
                                   img_name=self.img_name,
                                   seg_name=self.seg_name,
                                   output_dir=temp_dir,
                                   device=self.device)
            output_image = nib.load(output_file).get_fdata()
            np.testing.assert_allclose(np.sum(output_image), 34070)
            np.testing.assert_allclose(output_image.shape, (28, 25, 63, 1))


if __name__ == "__main__":
    unittest.main()
