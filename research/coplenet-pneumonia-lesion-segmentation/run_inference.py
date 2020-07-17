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
from glob import glob

import monai
import numpy as np
import torch
from monai.data import NiftiSaver
from monai.inferers import sliding_window_inference
from monai.transforms import AddChanneld, Compose, LoadNiftid, Orientationd, ToTensord

from coplenet import CopleNet

IMAGE_FOLDER = os.path.join(".", "images")
MODEL_FILE = os.path.join(".", "model", "coplenet_pretrained_monai_dict.pt")
OUTPUT_FOLDER = os.path.join(".", "output")  # writer will create this folder if it doesn't exist.


def main():
    images = sorted(glob(os.path.join(IMAGE_FOLDER, "case*.nii.gz")))
    val_files = [{"img": img} for img in images]

    # define transforms for image and segmentation
    infer_transforms = Compose(
        [
            LoadNiftid("img"),
            AddChanneld("img"),
            Orientationd("img", "SPL"),  # coplenet works on the plane defined by the last two axes
            ToTensord("img"),
        ]
    )
    test_ds = monai.data.Dataset(data=val_files, transform=infer_transforms)
    # sliding window inference need to input 1 image in every iteration
    data_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CopleNet().to(device)

    model.load_state_dict(torch.load(MODEL_FILE)["model_state_dict"])
    model.eval()

    with torch.no_grad():
        saver = NiftiSaver(output_dir=OUTPUT_FOLDER)
        for idx, val_data in enumerate(data_loader):
            print(f"Inference on {idx+1} of {len(data_loader)}")
            val_images = val_data["img"].to(device)
            # define sliding window size and batch size for windows inference
            slice_shape = np.ceil(np.asarray(val_images.shape[3:]) / 32) * 32
            roi_size = (20, int(slice_shape[0]), int(slice_shape[1]))
            sw_batch_size = 2
            val_outputs = sliding_window_inference(
                val_images, roi_size, sw_batch_size, model, 0.0, padding_mode="circular"
            )
            # val_outputs = (val_outputs.sigmoid() >= 0.5).float()
            val_outputs = val_outputs.argmax(dim=1, keepdim=True)
            saver.save_batch(val_outputs, val_data["img_meta_dict"])


if __name__ == "__main__":
    main()
