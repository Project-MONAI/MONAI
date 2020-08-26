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

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from DataPreprocessor import load_rg_data
from Dataset import RGDataset
from rggan import GenNet as G_NET
from torchvision.utils import make_grid

from monai import config
from monai.engines import default_make_latent
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadNiftid,
    NormalizeIntensityd,
    Resized,
    ShiftIntensityd,
    ThresholdIntensityd,
    ToTensord,
)
from monai.utils.misc import set_determinism


def run_evaluation(
    data_dir,
    rggan_saved_model,
    index,
    output_dir="./output",
    no_save=False,
    no_plot=True,
    seed=123,
    device_str="cuda:0",
    randomize_latents=True,
):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = torch.device(device_str)
    set_determinism(seed)

    # load images into array
    data_dict = load_rg_data(data_dir)

    # define transforms
    image_shape = (128, 128)
    eval_transforms = Compose(
        [
            LoadNiftid(keys=["base"], dtype=np.float64),
            Resized(keys=["base"], spatial_size=image_shape),
            ThresholdIntensityd(keys=["base"], threshold=-1000, above=True, cval=-1000),
            ThresholdIntensityd(keys=["base"], threshold=500, above=False, cval=500),
            ShiftIntensityd(keys=["base"], offset=1000),
            NormalizeIntensityd(
                keys=["base"],
                subtrahend=np.full(image_shape, 750.0),
                divisor=np.full(image_shape, 750.0),
                channel_wise=True,
            ),
            AddChanneld(keys="base"),
            ToTensord(keys=["base", "embedding"]),
        ]
    )

    # Create MONAI Dataset
    dataset = RGDataset(data_dict, eval_transforms, cache_num=0)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    # load generator
    logging.info("Loading RGGAN model: %s" % rggan_saved_model)
    gen_net = G_NET()
    gen_net.load_state_dict(torch.load(rggan_saved_model)["g_net"])
    gen_net.to(device)
    gen_net.eval()

    # create generator input
    data = dataset[index]
    latent_codes = default_make_latent(num_latents=1, latent_size=10, device=device)
    embedding = data["embedding"].to(device)
    random_bgs = data["base"]
    random_bgs = random_bgs.permute(1, 0, 2, 3).to(device)

    # execute generator forward
    scans, segs, impacts = [], [], []

    logging.info("Generating images with embedding: %s" % index)
    for base in random_bgs:
        if randomize_latents:
            latent_codes = default_make_latent(num_latents=1, latent_size=10, device=device)
        _, chest_ct, nodule_seg, gene_code_impact = gen_net.forward(latent_codes, embedding, base[None])
        scans.append(torch.squeeze(chest_ct))
        segs.append(nodule_seg)
        impacts.append(gene_code_impact)

    # save images
    img_affline = data["base_meta_dict"]["affine"]

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # create chest ct scans image grid
    chest_cts = torch.stack(scans)  # [10, 128, 128]
    chest_cts = torch.unsqueeze(chest_cts, dim=1)  # [10, 1, 128, 128]
    grid = make_grid(chest_cts.data, 5)
    ndarr = grid.mul_(750).add_(-250).clamp_(-1000, 1000).permute(2, 1, 0).cpu().numpy()
    chest_ct_out = ndarr[:, :, 0]

    # create nodule segments image grid
    nodule_segs = torch.stack(segs)
    nodule_segs = torch.squeeze(nodule_segs)
    nodule_segs = torch.unsqueeze(nodule_segs, dim=1)
    grid = make_grid(nodule_segs.data, 5)
    ndarr = grid.mul_(255).clamp_(0, 255).permute(2, 1, 0).cpu().numpy()
    nodule_seg_out = ndarr[:, :, 0]

    # create gene impact map image grid
    gene_impacts = torch.stack(impacts)
    gene_impacts = torch.squeeze(gene_impacts)
    gene_impacts = torch.unsqueeze(gene_impacts, dim=1)
    grid = make_grid(gene_impacts.data, 5)
    ndarr = grid.mul_(750).add_(-250).clamp_(-1000, 1000).permute(2, 1, 0).cpu().numpy()
    gene_impact_out = ndarr[:, :, 0]

    # create background bases image grid
    grid = make_grid(random_bgs.data, 5)
    ndarr = grid.mul_(750).add_(-250).clamp_(-1000, 1000).permute(2, 1, 0).cpu().numpy()
    bgs_out = ndarr[:, :, 0]

    if not no_save:
        print("Saving images in: %s" % output_dir)
        save_path = os.path.join(output_dir, f"Chest_CT_{index}_{seed}.nii.gz")
        nib.save(nib.Nifti1Image(chest_ct_out, img_affline), save_path)
        save_path = os.path.join(output_dir, f"Nodule_Masks_{index}_{seed}.nii.gz")
        nib.save(nib.Nifti1Image(nodule_seg_out, img_affline), save_path)
        save_path = os.path.join(output_dir, f"Gene_Impact_{index}_{seed}.nii.gz")
        nib.save(nib.Nifti1Image(gene_impact_out, img_affline), save_path)
        save_path = os.path.join(output_dir, f"Backgrounds_{index}_{seed}.nii.gz")
        nib.save(nib.Nifti1Image(bgs_out, img_affline), save_path)

    if not no_plot:
        figs, axs = plt.subplots(1, 4)
        axs[0].imshow(chest_ct_out)
        axs[0].set_title("Chest CT %d %d" % (index, seed))
        axs[1].imshow(nodule_seg_out)
        axs[1].set_title("Nodule Mask")
        axs[2].imshow(gene_impact_out)
        axs[2].set_title("Gene Code Impact")
        axs[3].imshow(bgs_out)
        axs[3].set_title("Background Images")
        plt.show()


def execute_cmdline():
    """
    Note: Developed, but not used right now.
    TODO: Decide what to do with this before PR
    """
    parser = argparse.ArgumentParser(
        prog="RGGAN Evaluation",
        description="Pass embedding and background image into generator to create chest CT scans, nodule masks, and gene fusion maps.",
        epilog="https://github.com/Project-MONAI/MONAI",
    )
    parser.add_argument("--index", help="Embedding from DS to evaluate.", default=1, type=int)
    parser.add_argument("--network", help="Path to saved RGGAN model .pth file.", default=None, type=str)
    parser.add_argument(
        "--input", help="TCIA NSCLC Radiogenomics data directory.", default="/nvdata/NSCLC-Ziyue", type=str
    )
    parser.add_argument("--noplot", help="Do not display plot.", default=False, type=bool)
    parser.add_argument("--nosave", help="Do not save images to disk.", default=True, type=bool)
    parser.add_argument("--output", help="Output folder for saved images.", default="/nvdata/output", type=str)
    parser.add_argument("--seed", help="Seed for random latent code.", default=22, type=int)
    parser.add_argument("--device", help="Torch.device to execute model.", default="cuda:0", type=str)
    args = parser.parse_args()


if __name__ == "__main__":
    run_evaluation(
        data_dir="/nvdata/NSCLC-Ziyue",
        rggan_saved_model="/nvdata/rggan/monai/net/g_synth_50.pth",
        output_dir="/nvdata/rggan/monai/net/g_synth_50",
        index=666,
        seed=121,
        no_save=False,
        no_plot=False,
        randomize_latents=True,
        device_str="cuda:0",
    )
