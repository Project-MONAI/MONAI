import argparse
import os

import nibabel as nib
import numpy as np

from monai.data import create_test_image_3d

root_path = os.path.dirname(__file__)
asset_path = os.path.join(root_path, "../assets")


def generate_data_sample(path, prefix):
    im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)

    n = nib.Nifti1Image(im, np.eye(4))
    nib.save(n, os.path.join(path, f"{prefix}_im.nii.gz"))

    n = nib.Nifti1Image(seg, np.eye(4))
    nib.save(n, os.path.join(path, f"{prefix}_seg.nii.gz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train-data-samples", type=int, default=40, help="Number of train data samples to generate")
    parser.add_argument("--n-test-data-samples", type=int, default=5, help="Number of test data samples to generate")
    args = parser.parse_args()

    print(f"Generating {args.n_train_data_samples} train data samples and {args.n_test_data_samples} test data samples")

    for i in range(args.n_train_data_samples):
        path = os.path.join(asset_path, f"train_data_samples/train_data_sample_{i}")
        os.makedirs(path, exist_ok=True)
        generate_data_sample(path, prefix=i)

    for i in range(args.n_test_data_samples):
        path = os.path.join(asset_path, f"test_data_samples/test_data_sample_{i}")
        os.makedirs(path, exist_ok=True)
        generate_data_sample(path, prefix=i)
