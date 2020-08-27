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

import datetime
import logging
import os
import random
import sys
from argparse import ArgumentParser

import dateutil.tz
import numpy as np
import torch

# RGGAN Imports
from RGDataPreprocessor import load_rg_data
from RGDataset import RGDataset
from RGModel import DiscNet, GenNet
from RGUtils import weights_init

# MONAI Framework
from monai import config
from monai.data import DataLoader
from monai.engines import GanTrainer, default_make_latent
from monai.engines.utils import GanKeys
from monai.handlers import CheckpointSaver, StatsHandler
from monai.inferers import Inferer
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadNiftid,
    NormalizeIntensityd,
    Resized,
    ShiftIntensityd,
    SqueezeDimd,
    ThresholdIntensityd,
    ToTensord,
)
from monai.utils.enums import InterpolateMode
from monai.utils.misc import set_determinism


def run_training(
    input_dir,
    output_dir,
    batch_size,
    num_epochs,
    cache_rate,
    workers,
    save_interval,
    seed,
    device,
    img_coeff,
    img_seg_coeff,
    img_seg_code_coeff,
    bg_reconst_coeff,
    g_lr,
    d_lr,
    g_n_feat,
    d_n_feat,
    embed_dim,
    latent_size,
    code_kernel,
):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # Required for Handler output
    device = torch.device(device)
    set_determinism(seed)

    # Init output data directories.
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    run_dir = os.path.join(output_dir, "radiogenomic-gan_%s" % (now.strftime("%Y_%m_%d_%H_%M_%S")))
    print("Using run directory: %s" % run_dir)

    # Create dictionary of data files.
    data_dict = load_rg_data(input_dir)

    # Define image transform pipeline.
    image_size = 128
    image_shape = [image_size, image_size]

    train_transforms = Compose(
        [
            LoadNiftid(keys=["image", "seg", "base"], dtype=np.float64),
            SqueezeDimd(keys=["rna_embedding"], dim=0),
            AddChanneld(keys=["image", "seg"]),
            ThresholdIntensityd(keys=["seg"], threshold=0.5),
            Resized(keys=["image", "seg", "base"], spatial_size=image_shape, mode=InterpolateMode.AREA),
            ThresholdIntensityd(keys=["image", "base"], threshold=-1000, above=True, cval=-1000),
            ThresholdIntensityd(keys=["image", "base"], threshold=500, above=False, cval=500),
            ShiftIntensityd(keys=["image", "base"], offset=1000),
            NormalizeIntensityd(
                keys=["image", "base"],
                subtrahend=np.full(image_shape, 750.0),
                divisor=np.full(image_shape, 750.0),
                channel_wise=True,
            ),
            NormalizeIntensityd(
                keys=["seg"], subtrahend=np.full(image_shape, 0.5), divisor=np.full(image_shape, 0.5), channel_wise=True
            ),
            ToTensord(keys=["image", "seg", "base"]),
        ]
    )

    # Create dataset and dataloader.
    dataset = RGDataset(data_dict, train_transforms, cache_rate=cache_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=workers)

    # Make a pointer to bg images from current discriminator training batch to calculate generator loss.
    minibatch_backgrounds: Optional[torch.Tensor] = None

    # Define prepare_batch for input into RGGAN G and D.
    def g_prepare_batch(batch_size, latent_size, device, batchdata):
        latent_codes = default_make_latent(batch_size, latent_size, device)
        embedding = batchdata["rna_embedding"].to(device)
        # select random background bases for this batch
        base_groups = batchdata["base"]
        curr_bgs = []
        for bases in base_groups:
            base_ix = random.randint(0, len(bases) - 1)
            curr_bgs.append(bases[base_ix][None])
        bases = torch.stack(curr_bgs).to(device)
        global minibatch_backgrounds
        minibatch_backgrounds = bases.detach()  # save pointer to bgs for loss func
        return (latent_codes, embedding, bases)

    def d_prepare_batch(batchdata, device):
        real_imgs = batchdata["image"].to(device)
        real_segs = batchdata["seg"].to(device)
        wrong_imgs = batchdata["w_image"].to(device)
        wrong_segs = batchdata["w_seg"].to(device)
        return (real_imgs, real_segs, wrong_imgs, wrong_segs)

    # Define inferer to call Generator forward() with input data.
    class GenInferer(Inferer):
        def __call__(self, inputs: torch.Tensor, network: torch.nn.Module):
            latent_codes, embedding, base = inputs
            return network(latent_codes, embedding, base)

    # Create and initialize RGGAN G and D networks.
    gen_net = GenNet(
        spatial_dim=2, img_features=g_n_feat, embed_dim=embed_dim, latent_size=latent_size, code_k_size=code_kernel
    ).to(device)
    gen_net.apply(weights_init)

    disc_net = DiscNet(spatial_dim=2, img_features=d_n_feat, embed_dim=embed_dim, code_k_size=code_kernel).to(device)
    disc_net.apply(weights_init)

    # Define optimizers.
    gen_opt = torch.optim.Adam(gen_net.parameters(), lr=g_lr, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(disc_net.parameters(), lr=d_lr, betas=(0.5, 0.999))

    # Define loss functions.
    real_labels = torch.ones(batch_size).to(device)
    fake_labels = torch.zeros(batch_size).to(device)
    logit_criterion = torch.nn.MSELoss()
    bg_reconst_criterion = torch.nn.L1Loss()

    def disc_loss(g_output, d_input):
        c_code, fake_imgs, fake_segs, _ = g_output
        real_imgs, real_segs, wrong_imgs, wrong_segs = d_input

        logits_real = disc_net(real_imgs, c_code.detach(), real_segs)
        logits_wcode = disc_net(wrong_imgs, c_code.detach(), wrong_segs)
        logits_wseg = disc_net(real_imgs, c_code.detach(), wrong_segs)
        logits_fake = disc_net(fake_imgs.detach(), c_code.detach(), fake_segs.detach())

        # gene code, image, segment realism
        loss_real = img_seg_code_coeff * logit_criterion(logits_real[0], real_labels)
        loss_wcode = img_seg_code_coeff * logit_criterion(logits_wcode[0], fake_labels)
        loss_wseg = img_seg_code_coeff * logit_criterion(logits_wseg[0], fake_labels)
        loss_fakes = img_seg_code_coeff * logit_criterion(logits_fake[0], fake_labels)

        # image realism
        if img_coeff > 0:
            loss_real += img_coeff * logit_criterion(logits_real[1], real_labels)
            loss_wcode += img_coeff * logit_criterion(logits_wcode[1], real_labels)
            loss_wseg += img_coeff * logit_criterion(logits_wseg[1], real_labels)
            loss_fakes += img_coeff * logit_criterion(logits_fake[1], fake_labels)

        # image segment realism
        if img_seg_coeff > 0:
            loss_real += img_seg_coeff * logit_criterion(logits_real[2], real_labels)
            loss_wcode += img_seg_coeff * logit_criterion(logits_wcode[2], real_labels)
            loss_wseg += img_seg_coeff * logit_criterion(logits_wseg[2], fake_labels)
            loss_fakes += img_seg_coeff * logit_criterion(logits_fake[2], fake_labels)

        return loss_real + loss_wcode + loss_wseg + loss_fakes

    bg_loss, pair_loss = None, None

    def gen_loss(g_output):
        global minibatch_backgrounds
        c_code, fake_imgs, fake_segs, fg_switches = g_output
        logits = disc_net(fake_imgs, c_code, fake_segs)
        img_seg_code_realism, img_realism, img_seg_realism = disc_net(fake_imgs, c_code, fake_segs)
        gen_loss = img_seg_code_coeff * logit_criterion(img_seg_code_realism, real_labels)

        if img_coeff > 0:
            "Image realism loss"
            gen_loss += img_coeff * logit_criterion(img_realism, real_labels)

        if img_seg_coeff > 0:
            "Image segment realism loss"
            gen_loss += img_seg_coeff * logit_criterion(img_seg_realism, real_labels)

        if bg_reconst_coeff > 0:
            "Background reconstruction loss. Penalize G modifications to the background."
            bg_mask = fake_segs.detach() < 0
            bg_fake = fake_imgs * bg_mask
            bg_img = minibatch_backgrounds * bg_mask
            bg_loss = bg_reconst_coeff * bg_reconst_criterion(bg_fake, bg_img)
            gen_loss += bg_loss

        return gen_loss

    # Define training event handlers.
    handlers = [
        StatsHandler(
            name="training_loss",
            output_transform=lambda x: {GanKeys.GLOSS: x[GanKeys.GLOSS], GanKeys.DLOSS: x[GanKeys.DLOSS]},
        ),
        CheckpointSaver(
            save_dir=run_dir,
            save_dict={"g_net": gen_net, "d_net": disc_net},
            epoch_level=True,
            save_final=True,
            save_interval=save_interval,
        ),
    ]

    # Create GanTrainer.
    trainer = GanTrainer(
        device,
        num_epochs,
        dataloader,
        gen_net,
        gen_opt,
        gen_loss,
        disc_net,
        disc_opt,
        disc_loss,
        d_prepare_batch=d_prepare_batch,
        g_prepare_batch=g_prepare_batch,
        g_inferer=GenInferer(),
        g_update_latents=False,
        latent_size=latent_size,
        key_train_metric=None,
        train_handlers=handlers,
    )

    trainer.run()


def execute_cmdline():
    parser = ArgumentParser(prog="MONAI RadiogenomicGAN Training Script")
    parser.add_argument(
        "--device", type=str, default="cuda:0", dest="device", help="device type string. e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument("--ep", type=int, default=500, dest="num_epochs")
    parser.add_argument("--bs", type=int, default=16, dest="batch_size")
    parser.add_argument("--seed", type=int, default=12345, dest="seed")
    parser.add_argument("--input", type=str, default="./data", dest="input_dir")
    parser.add_argument("--output", type=str, default="./ModelOut", dest="output_dir")
    parser.add_argument(
        "--save_interval", type=int, default=50, dest="save_interval", help="Save checkpoints every N epochs.",
    )
    parser.add_argument("--g_lr", type=float, default=0.0001, dest="g_lr", help="Loss rate for G ADAM optimizer")
    parser.add_argument("--d_lr", type=float, default=0.0001, dest="d_lr", help="Loss rate for D ADAM optimizer")
    parser.add_argument("--g_n_feat", type=int, default=32, dest="g_n_feat", help="Number of image feature maps for G")
    parser.add_argument("--d_n_feat", type=int, default=64, dest="d_n_feat", help="Number of image feature maps for D")
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=128,
        dest="embed_dim",
        help="Size of g_encoder_net output description embedding.",
    )
    parser.add_argument("--g_ls", type=int, default=10, dest="latent_size", help="Size of latent input for GenNet.")
    parser.add_argument(
        "--code_kernel", type=int, default=8, dest="code_kernel", help="Size of kernel for code evaluation."
    )
    parser.add_argument("--c_i", type=float, default=8.0, help="Image loss coeff", dest="img_coeff")
    parser.add_argument("--c_is", type=float, default=1.0, help="Image-Segment loss coeff", dest="img_seg_coeff")
    parser.add_argument(
        "--c_isc", type=float, default=10.0, help="Image-Segment-Code loss coeff", dest="img_seg_code_coeff"
    )
    parser.add_argument(
        "--c_bg", type=float, default=100.0, help="BgReconstruction loss coeff", dest="bg_reconst_coeff"
    )
    parser.add_argument(
        "--workers", type=int, default=4, dest="workers", help="Number of processing units for DataLoader"
    )
    parser.add_argument("--cr", type=float, default=1.0, dest="cache_rate")
    args = parser.parse_args()

    input_dict = vars(args)
    print(input_dict)
    run_training(**input_dict)


if __name__ == "__main__":
    print("Start RGGAN Training Script.")
    execute_cmdline()
    print("End RGGAN Training Script.")
