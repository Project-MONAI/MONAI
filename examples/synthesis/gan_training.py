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
"""
MONAI Generative Adversarial Networks Workflow Example
    Sample script using MONAI to train a GAN to synthesize images from a latent code.

## Get the dataset
    MedNIST.tar.gz link: https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz
    Extract tarball and set input_dir variable. GAN script trains using hand CT scan jpg images.

    Dataset information available in MedNIST Tutorial
    https://github.com/Project-MONAI/Tutorials/blob/master/mednist_tutorial.ipynb
"""

import logging
import os
import sys

import torch

import monai
from monai.apps.utils import download_and_extract
from monai.data import CacheDataset, DataLoader, png_writer
from monai.engines import GanTrainer
from monai.engines.utils import GanKeys as Keys
from monai.engines.utils import default_make_latent as make_latent
from monai.handlers import CheckpointSaver, StatsHandler
from monai.networks import normal_init
from monai.networks.nets import Discriminator, Generator
from monai.transforms import (
    AddChannelD,
    Compose,
    LoadPNGD,
    RandFlipD,
    RandRotateD,
    RandZoomD,
    ScaleIntensityD,
    ToTensorD,
)
from monai.utils.misc import set_determinism


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    set_determinism(12345)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load real data
    mednist_url = "https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz?dl=1"
    md5_value = "0bc7306e7427e00ad1c5526a6677552d"
    extract_dir = "data"
    tar_save_path = os.path.join(extract_dir, "MedNIST.tar.gz")
    download_and_extract(mednist_url, tar_save_path, extract_dir, md5_value)
    hand_dir = os.path.join(extract_dir, "MedNIST", "Hand")
    real_data = [{"hand": os.path.join(hand_dir, filename)} for filename in os.listdir(hand_dir)]

    # define real data transforms
    train_transforms = Compose(
        [
            LoadPNGD(keys=["hand"]),
            AddChannelD(keys=["hand"]),
            ScaleIntensityD(keys=["hand"]),
            RandRotateD(keys=["hand"], range_x=15, prob=0.5, keep_size=True),
            RandFlipD(keys=["hand"], spatial_axis=0, prob=0.5),
            RandZoomD(keys=["hand"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
            ToTensorD(keys=["hand"]),
        ]
    )

    # create dataset and dataloader
    real_dataset = CacheDataset(real_data, train_transforms)
    batch_size = 300
    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    # define function to process batchdata for input into discriminator
    def prepare_batch(batchdata):
        """
        Process Dataloader batchdata dict object and return image tensors for D Inferer
        """
        return batchdata["hand"]

    # define networks
    disc_net = Discriminator(
        in_shape=(1, 64, 64), channels=(8, 16, 32, 64, 1), strides=(2, 2, 2, 2, 1), num_res_units=1, kernel_size=5
    ).to(device)

    latent_size = 64
    gen_net = Generator(
        latent_shape=latent_size, start_shape=(latent_size, 8, 8), channels=[32, 16, 8, 1], strides=[2, 2, 2, 1]
    )

    # initialize both networks
    disc_net.apply(normal_init)
    gen_net.apply(normal_init)

    # input images are scaled to [0,1] so enforce the same of generated outputs
    gen_net.conv.add_module("activation", torch.nn.Sigmoid())
    gen_net = gen_net.to(device)

    # create optimizers and loss functions
    learning_rate = 2e-4
    betas = (0.5, 0.999)
    disc_opt = torch.optim.Adam(disc_net.parameters(), learning_rate, betas=betas)
    gen_opt = torch.optim.Adam(gen_net.parameters(), learning_rate, betas=betas)

    disc_loss_criterion = torch.nn.BCELoss()
    gen_loss_criterion = torch.nn.BCELoss()
    real_label = 1
    fake_label = 0

    def discriminator_loss(gen_images, real_images):
        """
        The discriminator loss is calculated by comparing D
        prediction for real and generated images.

        """
        real = real_images.new_full((real_images.shape[0], 1), real_label)
        gen = gen_images.new_full((gen_images.shape[0], 1), fake_label)

        realloss = disc_loss_criterion(disc_net(real_images), real)
        genloss = disc_loss_criterion(disc_net(gen_images.detach()), gen)

        return (genloss + realloss) / 2

    def generator_loss(gen_images):
        """
        The generator loss is calculated by determining how realistic
        the discriminator classifies the generated images.

        """
        output = disc_net(gen_images)
        cats = output.new_full(output.shape, real_label)
        return gen_loss_criterion(output, cats)

    # initialize current run dir
    run_dir = "model_out"
    print("Saving model output to: %s " % run_dir)

    # create workflow handlers
    handlers = [
        StatsHandler(
            name="batch_training_loss",
            output_transform=lambda x: {Keys.GLOSS: x[Keys.GLOSS], Keys.DLOSS: x[Keys.DLOSS]},
        ),
        CheckpointSaver(
            save_dir=run_dir,
            save_dict={"g_net": gen_net, "d_net": disc_net},
            save_interval=10,
            save_final=True,
            epoch_level=True,
        ),
    ]

    # define key metric
    key_train_metric = None

    # create adversarial trainer
    disc_train_steps = 5
    num_epochs = 50

    trainer = GanTrainer(
        device,
        num_epochs,
        real_dataloader,
        gen_net,
        gen_opt,
        generator_loss,
        disc_net,
        disc_opt,
        discriminator_loss,
        d_prepare_batch=prepare_batch,
        d_train_steps=disc_train_steps,
        latent_shape=latent_size,
        key_train_metric=key_train_metric,
        train_handlers=handlers,
    )

    # run GAN training
    trainer.run()

    # Training completed, save a few random generated images.
    print("Saving trained generator sample output.")
    test_img_count = 10
    test_latents = make_latent(test_img_count, latent_size).to(device)
    fakes = gen_net(test_latents)
    for i, image in enumerate(fakes):
        filename = "gen-fake-final-%d.png" % i
        save_path = os.path.join(run_dir, filename)
        img_array = image[0].cpu().data.numpy()
        png_writer.write_png(img_array, save_path, scale=255)


if __name__ == "__main__":
    main()
