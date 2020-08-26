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
import random
import sys

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


def main():
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # Required for Handler output
    device = torch.device("cuda:1")
    set_determinism(12345)

    # Define data directories.
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    output_data_dir = "ModelOut/radiogenomic-gan_%s" % (now.strftime("%Y_%m_%d_%H_%M_%S"))
    input_data_dir = "/home/gagan/code/MONAI/research/radiogenomic-gan-nodule-synthesis/data"

    print("Input Data Dir: %s" % input_data_dir)
    print("Output Data Dir: %s" % output_data_dir)

    # Create dictionary of data files.
    data_dict = load_rg_data(input_data_dir)

    # Define image transform pipeline.
    image_size = 128
    image_shape = [image_size, image_size]

    train_transforms = Compose(
        [
            LoadNiftid(keys=["image", "seg", "base"], dtype=np.float64),
            SqueezeDimd(keys=["embedding"], dim=0),
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
    cache_max: double = 3
    dataset = RGDataset(data_dict, train_transforms, cache_max)
    batch_size = 16
    num_workers: int = 5
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=num_workers)

    # Store a pointer to base images from current batch for calculating generator loss.
    minibatch_backgrounds: Optional[torch.Tensor] = None

    # Define prepare_batch for input into RGGAN G and D.
    def g_prepare_batch(batch_size, latent_size, device, batchdata):
        global minibatch_backgrounds
        latent_codes = default_make_latent(batch_size, latent_size, device)
        embedding = batchdata["embedding"].to(device)
        # select random background bases for this batch
        base_groups = batchdata["base"]
        curr_bgs = []
        for bases in base_groups:
            base_ix = random.randint(0, len(bases) - 1)
            curr_bgs.append(bases[base_ix][None])
        bases = torch.stack(curr_bgs).to(device)
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
    gen_net = GenNet().to(device)
    gen_net.apply(weights_init)

    disc_net = DiscNet().to(device)
    disc_net.apply(weights_init)

    # Define optimizers.
    gen_opt = torch.optim.Adam(gen_net.parameters(), lr=0.0001, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(disc_net.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Define loss functions.
    real_labels = torch.ones(batch_size).to(device)
    fake_labels = torch.zeros(batch_size).to(device)
    logit_criterion = torch.nn.MSELoss()
    bg_reconst_criterion = torch.nn.L1Loss()

    img_seg_code_loss_coeff = 10.0
    img_loss_coeff = 8.0
    img_seg_loss_coeff = 1.0
    bg_reconst_loss_coeff = 100.0

    def disc_loss(g_output, d_input):
        c_code, fake_imgs, fake_segs, _ = g_output
        real_imgs, real_segs, wrong_imgs, wrong_segs = d_input

        logits_real = disc_net(real_imgs, c_code.detach(), real_segs)
        logits_wcode = disc_net(wrong_imgs, c_code.detach(), wrong_segs)
        logits_wseg = disc_net(real_imgs, c_code.detach(), wrong_segs)
        logits_fake = disc_net(fake_imgs.detach(), c_code.detach(), fake_segs.detach())

        # gene code, image, segment realism
        loss_real = img_seg_code_loss_coeff * logit_criterion(logits_real[0], real_labels)
        loss_wcode = img_seg_code_loss_coeff * logit_criterion(logits_wcode[0], fake_labels)
        loss_wseg = img_seg_code_loss_coeff * logit_criterion(logits_wseg[0], fake_labels)
        loss_fakes = img_seg_code_loss_coeff * logit_criterion(logits_fake[0], fake_labels)

        # image realism
        if img_loss_coeff > 0:
            loss_real += img_loss_coeff * logit_criterion(logits_real[1], real_labels)
            loss_wcode += img_loss_coeff * logit_criterion(logits_wcode[1], real_labels)
            loss_wseg += img_loss_coeff * logit_criterion(logits_wseg[1], real_labels)
            loss_fakes += img_loss_coeff * logit_criterion(logits_fake[1], fake_labels)

        # image segment realism
        if img_seg_loss_coeff > 0:
            loss_real += img_seg_loss_coeff * logit_criterion(logits_real[2], real_labels)
            loss_wcode += img_seg_loss_coeff * logit_criterion(logits_wcode[2], real_labels)
            loss_wseg += img_seg_loss_coeff * logit_criterion(logits_wseg[2], fake_labels)
            loss_fakes += img_seg_loss_coeff * logit_criterion(logits_fake[2], fake_labels)

        return loss_real + loss_wcode + loss_wseg + loss_fakes

    bg_loss, pair_loss = None, None

    def gen_loss(g_output):
        global minibatch_backgrounds
        c_code, fake_imgs, fake_segs, fg_switches = g_output
        logits = disc_net(fake_imgs, c_code, fake_segs)
        img_seg_code_realism, img_realism, img_seg_realism = disc_net(fake_imgs, c_code, fake_segs)
        gen_loss = img_seg_code_loss_coeff * logit_criterion(img_seg_code_realism, real_labels)

        if img_loss_coeff > 0:
            "Image realism loss"
            gen_loss += img_loss_coeff * logit_criterion(img_realism, real_labels)

        if img_seg_loss_coeff > 0:
            "Image segment realism loss"
            gen_loss += img_seg_loss_coeff * logit_criterion(img_seg_realism, real_labels)

        if bg_reconst_loss_coeff > 0:
            "Background reconstruction loss. Penalize G modifications to the background."
            bg_mask = fake_segs.detach() < 0
            bg_fake = fake_imgs * bg_mask
            bg_img = minibatch_backgrounds * bg_mask
            bg_loss = bg_reconst_loss_coeff * bg_reconst_criterion(bg_fake, bg_img)
            gen_loss += bg_loss

        return gen_loss

    # Define training event handlers.
    checkpoint_save_interval = 50

    handlers = [
        StatsHandler(
            name="training_loss",
            output_transform=lambda x: {GanKeys.GLOSS: x[GanKeys.GLOSS], GanKeys.DLOSS: x[GanKeys.DLOSS]},
        ),
        CheckpointSaver(
            save_dir=output_data_dir,
            save_dict={"g_net": gen_net, "d_net": disc_net},
            epoch_level=True,
            save_final=True,
            save_interval=checkpoint_save_interval,
        ),
    ]

    # Create GanTrainer.
    latent_size = 10
    num_epochs = 500

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

    print("END MAIN")


if __name__ == "__main__":
    main()
