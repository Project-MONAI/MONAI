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
    https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/mednist_tutorial.ipynb.
"""

import os
import sys
import logging
import torch

import monai
from monai.utils.misc import set_determinism, create_run_dir
from monai.data import CacheDataset, DataLoader, png_writer
from monai.networks.nets import Generator, Discriminator
from monai.networks import normal_init
from monai.engines import AdversarialTrainer
from monai.handlers import StatsHandler, CheckpointSaver
from monai.transforms import (
    Compose,
    LoadPNGD,
    AddChannelD,
    ScaleIntensityD,
    RandRotateD,
    RandFlipD,
    RandZoomD,
    ToTensorD,
)


def save_generator_fakes(run_folder, checkpoint, g_output_tensor):
    for i, image in enumerate(g_output_tensor):
        filename = "gen-fake-%s-%d.png" % (checkpoint, i)
        save_path = os.path.join(run_folder, filename)
        img_array = image[0].cpu().data.numpy()
        png_writer.write_png(img_array, save_path, scale=255)


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    set_determinism(12345)
    device = torch.device("cuda:0")

    # load real data
    input_dir = "/shahinaaz/nvdata/MedNIST/Hand"
    real_data = [{"hand": os.path.join(input_dir, filename)} for filename in os.listdir(input_dir)]

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

    # define function to process batchdata for input into generator
    def prepare_batch(batchdata):
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

    # Loss and Optimizers
    learning_rate = 2e-4
    betas = (0.5, 0.999)
    disc_opt = torch.optim.Adam(disc_net.parameters(), learning_rate, betas=betas)
    gen_opt = torch.optim.Adam(gen_net.parameters(), learning_rate, betas=betas)

    disc_loss = torch.nn.BCELoss()
    gen_loss = torch.nn.BCELoss()
    real_label = 1
    fake_label = 0

    def discriminator_loss(gen_images, real_images):
        """
        The discriminator loss if calculated by comparing its
        prediction for real and generated images.

        """
        real = real_images.new_full((real_images.shape[0], 1), real_label)
        gen = gen_images.new_full((gen_images.shape[0], 1), fake_label)

        realloss = disc_loss(disc_net(real_images), real)
        genloss = disc_loss(disc_net(gen_images.detach()), gen)

        return (realloss + genloss) / 2

    def generator_loss(input):
        """
        The generator loss is calculated by determining how well
        the discriminator was fooled by the generated images.

        """
        output = disc_net(input)
        cats = output.new_full(output.shape, real_label)
        return gen_loss(output, cats)

    # initialize current run dir
    run_dir = create_run_dir("./ModelOut")
    print("Saving model output to: %s " % run_dir)

    # create workflow handlers
    handlers = [
        StatsHandler(name="train_loss"),
        CheckpointSaver(
            save_dir=run_dir,
            save_dict={"g_net": gen_net, "d_net": disc_net},
            save_interval=10,
            save_final=True,
            epoch_level=True,
        ),
    ]

    # create adversarial trainer
    key_train_metric = None  # TODO: Make FID Monai Metric to evaluate generator performance
    disc_train_steps = 5
    num_epochs = 50

    trainer = AdversarialTrainer(
        device,
        num_epochs,
        real_dataloader,
        gen_net,
        gen_opt,
        generator_loss,
        disc_net,
        disc_opt,
        discriminator_loss,
        d_train_steps=disc_train_steps,
        latent_shape=latent_size,
        prepare_batch=prepare_batch,
        key_train_metric=key_train_metric,
        train_handlers=handlers,
    )

    # Run Training
    trainer.run()

    # Save a few randomly generated images. Hopefully most images will have four fingers and a thumb as expected
    # (assuming polydactyl examples were not present in large numbers in the dataset).
    print("Saving trained generator sample output.")
    # TODO: turn into gan_save_fakes handler
    test_img_count = 10
    test_latents = torch.randn(test_img_count, latent_size).to(device)
    save_generator_fakes(run_dir, "final", gen_net(test_latents))


if __name__ == "__main__":
    main()
