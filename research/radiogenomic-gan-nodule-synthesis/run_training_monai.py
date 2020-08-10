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
import sys

import cv2
import dateutil.tz
import numpy as np
import torch

# RGGAN Custom Libraries
from DataPreprocessor import load_rg_data
from Dataset import RGDataset
from ReparameterizationTrick import ReparameterizationApply, ReparameterizationRestore, ReparameterizationUpdate
from rggan_model import D_NET, G_NET
from rggan_utils import weights_init
from Transforms import CropWithBoundingBoxd

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
    device = torch.device("cuda:0")
    set_determinism(12345)

    # Define data directories.
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    output_data_dir = "ModelOut/radiogenomic-gan_%s" % (now.strftime("%Y_%m_%d_%H_%M_%S"))
    input_data_dir = "/home/gagan/code/MONAI/research/radiogenomic-gan-nodule-synthesis/data"

    print("Input Data Dir: %s" % input_data_dir)
    print("Output Data Dir: %s" % output_data_dir)

    # Load data filepath dictionary.
    data_dict = load_rg_data(input_data_dir)

    # Define image transform pipeline.
    image_size = 128
    image_shape = [image_size, image_size]

    train_transforms = Compose(
        [
            LoadNiftid(keys=["image", "seg", "base"], dtype=np.float64),
            SqueezeDimd(keys=["embedding"], dim=0),
            CropWithBoundingBoxd(keys=["image", "seg"], bbox_key="bbox"),
            AddChanneld(keys=["image", "seg", "base"]),
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
    cache_max: int = 25
    dataset = RGDataset(data_dict, train_transforms, cache_max)
    batch_size = 16
    num_workers: int = 5
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=num_workers)

    # Store a pointer to background base images from current batch to calculate generator loss.
    background_base: Optional[torch.Tensor] = None

    # Define prepare_batch for input into RGGAN G and D.
    def g_prepare_batch(batch_size, latent_size, device, batchdata):
        latent_codes = default_make_latent(batch_size, latent_size, device)
        embedding = batchdata["embedding"].to(device)
        base = batchdata["base"].to(device)  # todo - process filename in transform chain
        global background_base
        background_base = base
        return (latent_codes, embedding, base)

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
    gen_net = G_NET().to(device)
    gen_net.apply(weights_init)
    disc_net = D_NET().to(device)
    disc_net.apply(weights_init)

    # Define optimizers.
    gen_opt = torch.optim.Adam(gen_net.parameters(), lr=0.0001, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(disc_net.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Define loss functions.
    real_labels = torch.FloatTensor(batch_size).fill_(1).to(device)
    fake_labels = torch.FloatTensor(batch_size).fill_(0).to(device)
    criterion = torch.nn.MSELoss()
    rc_criterion = torch.nn.L1Loss()

    ITS_LOSS_COEFF = 10.0
    I_LOSS_COEFF = 8.0
    IS_LOSS_COEFF = 1.0
    RC_LOSS_COEFF = 100.0

    def disc_loss(g_output, d_input):
        c_code, fake_imgs, fake_segs, _ = g_output
        real_imgs, real_segs, wrong_imgs, wrong_segs = d_input

        logits_real = disc_net(real_imgs, c_code.detach(), real_segs)
        logits_wcode = disc_net(wrong_imgs, c_code.detach(), wrong_segs)
        logits_wseg = disc_net(real_imgs, c_code.detach(), wrong_segs)
        logits_fake = disc_net(fake_imgs.detach(), c_code.detach(), fake_segs.detach())

        loss_real = ITS_LOSS_COEFF * criterion(logits_real[0], real_labels)
        loss_wcode = ITS_LOSS_COEFF * criterion(logits_wcode[0], fake_labels)
        loss_wseg = ITS_LOSS_COEFF * criterion(logits_wseg[0], fake_labels)
        loss_fakes = ITS_LOSS_COEFF * criterion(logits_fake[0], fake_labels)

        if I_LOSS_COEFF > 0:
            loss_real += I_LOSS_COEFF * criterion(logits_real[1], real_labels)
            loss_wcode += I_LOSS_COEFF * criterion(logits_wcode[1], real_labels)
            loss_wseg += I_LOSS_COEFF * criterion(logits_wseg[1], real_labels)
            loss_fakes += I_LOSS_COEFF * criterion(logits_fake[1], fake_labels)

        if IS_LOSS_COEFF > 0:
            loss_real += IS_LOSS_COEFF * criterion(logits_real[2], real_labels)
            loss_wcode += IS_LOSS_COEFF * criterion(logits_wcode[2], real_labels)
            loss_wseg += IS_LOSS_COEFF * criterion(logits_wseg[2], fake_labels)
            loss_fakes += IS_LOSS_COEFF * criterion(logits_fake[2], fake_labels)

        return loss_real + loss_wcode + loss_wseg + loss_fakes

    bg_loss, pair_loss = None, None

    def gen_loss(g_output):
        global background_base
        c_code, fake_imgs, fake_segs, fg_switches = g_output
        feedback = disc_net(fake_imgs, c_code, fake_segs)
        gen_loss = ITS_LOSS_COEFF * criterion(feedback[0], real_labels)

        if I_LOSS_COEFF > 0:
            gen_loss += I_LOSS_COEFF * criterion(feedback[1], real_labels)

        if IS_LOSS_COEFF > 0:
            gen_loss += IS_LOSS_COEFF * criterion(feedback[2], real_labels)

        if RC_LOSS_COEFF > 0:
            temp_seg = fake_segs.detach()
            temp_mask = (temp_seg < 0).type(torch.FloatTensor)
            erode_mask = temp_mask.permute(0, 2, 3, 1).data.cpu().numpy()
            kernel = np.ones((5, 5), np.uint8)
            for index in range(batch_size):
                erode_mask[index, :, :, 0] = cv2.erode(erode_mask[index, :, :, 0], kernel, iterations=1)
                # TODO: replace cv2 erode() call
            bg_mask = torch.Tensor(erode_mask).to(device)
            bg_mask = bg_mask.permute(0, 3, 1, 2)
            bg_fake = fake_imgs * bg_mask
            bg_img = background_base * bg_mask
            bg_loss = RC_LOSS_COEFF * rc_criterion(bg_fake, bg_img)

            gen_loss += bg_loss

        return gen_loss

    # Define training event handlers.
    checkpoint_save_interval = 50

    handlers = [
        ReparameterizationUpdate(network=gen_net, update_interval=1, epoch_only=False),
        StatsHandler(
            name="training_loss",
            output_transform=lambda x: {GanKeys.GLOSS: x[GanKeys.GLOSS], GanKeys.DLOSS: x[GanKeys.DLOSS]},
        ),
        ReparameterizationApply(
            network=gen_net, epoch_level=True, save_final=True, save_interval=checkpoint_save_interval,
        ),
        CheckpointSaver(
            save_dir=output_data_dir,
            save_dict={"g_net": gen_net, "d_net": disc_net},
            epoch_level=True,
            save_final=True,
            save_interval=checkpoint_save_interval,
        ),
        ReparameterizationRestore(network=gen_net, epoch_level=True, save_interval=checkpoint_save_interval,),
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

    print("# start training")
    trainer.run()

    print("END MAIN")


if __name__ == "__main__":
    main()
