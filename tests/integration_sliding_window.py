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
import sys
import tempfile

import nibabel as nib
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


def run_test(batch_size=2, device=torch.device("cpu:0")):

    im, seg = create_test_image_3d(25, 28, 63, rad_max=10, noise_max=1, num_objs=4, num_seg_classes=1)
    input_shape = im.shape
    img_name = make_nifti_image(im)
    seg_name = make_nifti_image(seg)
    ds = NiftiDataset([img_name], [seg_name], transform=AddChannel(), seg_transform=AddChannel(), image_only=False)
    loader = DataLoader(ds, batch_size=1, pin_memory=torch.cuda.is_available())

    net = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(4, 8, 16, 32),
        strides=(2, 2, 2),
        num_res_units=2,
    )
    roi_size = (16, 32, 48)
    sw_batch_size = batch_size

    def _sliding_window_processor(_engine, batch):
        net.eval()
        img, seg, meta_data = batch
        with torch.no_grad():
            seg_probs = sliding_window_inference(img, roi_size, sw_batch_size, net, device)
            return predict_segmentation(seg_probs)

    infer_engine = Engine(_sliding_window_processor)

    with tempfile.TemporaryDirectory() as temp_dir:
        SegmentationSaver(output_path=temp_dir, output_ext='.nii.gz', output_postfix='seg',
                          batch_transform=lambda x: x[2]).attach(infer_engine)

        infer_engine.run(loader)

        basename = os.path.basename(img_name)[:-len('.nii.gz')]
        saved_name = os.path.join(temp_dir, basename, '{}_seg.nii.gz'.format(basename))
        # get spatial dimensions shape, the saved nifti image format: HWDC
        testing_shape = nib.load(saved_name).get_fdata().shape[:-1]

    if os.path.exists(img_name):
        os.remove(img_name)
    if os.path.exists(seg_name):
        os.remove(seg_name)
    if testing_shape != input_shape:
        print('testing shape: {} does not match input shape: {}.'.format(testing_shape, input_shape))
        return False
    return True


if __name__ == "__main__":
    result = run_test()
    sys.exit(0 if result else 1)
