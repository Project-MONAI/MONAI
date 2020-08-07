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
import dateutil.tz

import numpy as np
import torch

from monai import config
from monai.data import DataLoader
from monai.engines import GanTrainer, default_make_latent
from monai.inferers import Inferer
from monai.utils.misc import set_determinism
from monai.utils.enums import InterpolateMode
from monai.transforms import (
    Compose,
    LoadNiftid,
    Resized,
    AddChanneld,
    ThresholdIntensityd,
    ShiftIntensityd,
    NormalizeIntensityd,
    ToTensord,
)

from DataPreprocessor import load_rg_data
from Dataset import RGDataset
from Transforms import CropWithBoundingBoxd
from rggan_model import G_NET, D_NET
from rggan_utils import weights_init


def main():
    config.print_config()
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
            LoadNiftid(keys=["image", "seg"], dtype=np.float64),
            CropWithBoundingBoxd(keys=["image", "seg"], bbox_key="bbox"),
            AddChanneld(keys=["image", "seg"]),
            ThresholdIntensityd(keys=["seg"], threshold=0.5),
            Resized(keys=["image", "seg"], spatial_size=image_shape, mode=InterpolateMode.AREA),
            ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
            ThresholdIntensityd(keys=["image"], threshold=500, above=False, cval=500),
            ShiftIntensityd(keys=["image"], offset=1000),
            NormalizeIntensityd(
                keys=["image"],
                subtrahend=np.full(image_shape, 750.0),
                divisor=np.full(image_shape, 750.0),
                channel_wise=True,
            ),
            NormalizeIntensityd(
                keys=["seg"], subtrahend=np.full(image_shape, 0.5), divisor=np.full(image_shape, 0.5), channel_wise=True
            ),
            ToTensord(keys=["image", "seg"]),
        ]
    )

    # Create dataset and dataloader.
    cache_max: int = 25
    dataset = RGDataset(data_dict, train_transforms, cache_max)
    batch_size = 16
    num_workers: int = 5
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=num_workers)

    # Define prepare_batch for input into RGGAN G and D.
    def g_prepare_batch(batch_size, latent_size, batchdata):
        latent_codes = default_make_latent(batch_size, latent_size)
        embedding = batchdata["embedding"]
        base = batchdata["base"]  # todo - process filename in transform chain
        return (latent_codes.data, embedding.data, base)

    def d_prepare_batch(batchdata):
        real_imgs = batchdata["image"]
        real_segs = batchdata["seg"]
        wrong_imgs = batchdata["w_image"]
        wrong_segs = batchdata["w_seg"]
        return (real_imgs, real_segs, wrong_imgs, wrong_segs)

    # Define inferer to call Generator forward() with input data.
    class GenInferer(Inferer):
        def __call__(self, inputs: torch.Tensor, network: torch.nn.Module):
            latent_codes, embedding, base = inputs
            return network(latent_codes, embedding, base)

    # Create and initialize RGGAN G and D networks.
    gen_net = G_NET()
    gen_net.apply(weights_init)
    disc_net = D_NET()
    disc_net.apply(weights_init)

    # Define optimizers.
    gen_opt = torch.optim.Adam(gen_net.parameters(), lr=0.0001, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(disc_net.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Define loss functions.
    real_labels = 1
    fake_labels = 0
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

    def gen_loss(g_output):
        c_code, fake_imgs, fake_segs, fg_switches = g_output
        feedback = disc_net(fake_imgs, c_code, fake_segs)
        gen_loss = ITS_LOSS_COEFF * criterion(feedback[0], real_labels)

        if I_LOSS_COEFF > 0:
            gen_loss += I_LOSS_COEFF * criterion(feedback[1], real_labels)

        if IS_LOSS_COEFF > 0:
            gen_loss += IS_LOSS_COEFF * criterion(feedback[2], real_labels)

        if RC_LOSS_COEFF > 0:
            temp_seg = fake_segs[0].detach()
            temp_mask = (temp_seg < 0).type(torch.FloatTensor)
            erode_mask = temp_mask.permute(0, 2, 3, 1).data.cpu().numpy()
            kernel = np.ones(5, 5, np.uint8)
            for index in range(batch_size):
                erode_mask[index, :, :, 0] = cv2.erode(erode_mask[index, :, :, 0], kernel, iterations=1)
                # TODO: replace cv2 erode() call
            bg_mask = torch.Tensor(erode_mask)
            bg_mask = bg_mask.permute(0, 3, 1, 2)
            bg_fake = fake_imgs * bg_mask
            bg_img = fg_switches * bg_mask
            bg_loss = RC_LOSS_COEFF * self.rc_criterion(bg_fake, bg_img)

            gen_loss += bg_loss

        return gen_loss

    # Define training event handlers.
    handlers = []

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
        g_update_latents=True,
        latent_shape=latent_size,
        key_train_metric=None,
        train_handlers=handlers,
    )

    print('# start training')
    trainer.run()

    print("END MAIN")


if __name__ == "__main__":
    main()
