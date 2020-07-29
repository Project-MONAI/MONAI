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
MONAI Generative Adversarial Networks GAN Example
    Sample script using MONAI library to train generator with MedNIST CT Hands Dataset

## Get the dataset
    https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz

    Description: 
    https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/mednist_tutorial.ipynb.
"""

import os
import sys
import logging
import torch
import matplotlib.pyplot as plt

import monai
from monai.utils.misc import set_determinism, create_run_dir
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import Generator, Discriminator
from monai.networks import normal_init
from monai.engines import AdversarialTrainer
from monai.handlers import StatsHandler, CheckpointSaver
from monai.transforms import (
    LoadPNG,
    Compose,
    AddChannel,
    ScaleIntensity,
    ToTensor,
    RandRotate,
    RandFlip,
    RandZoom,
)


def save_images(run_folder, checkpoint, images):
    for i, image in enumerate(images):
        # img = Image.fromarray(image[0].cpu().data.numpy())
        savepath = os.path.join(run_folder, "output_%s_%d" % (checkpoint, i))
        # img.save(savepath, "TIFF")
        plt.figure()
        plt.imshow(image[0].cpu().data.numpy(), cmap="gray")
        plt.savefig(savepath)


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    set_determinism(12345)
    device = torch.device("cuda:0")

    # load real data
    MedNIST_Hand_Dir = "/shahinaaz/nvdata/MedNIST/Hand"
    hands = [os.path.join(MedNIST_Hand_Dir, filename) for filename in os.listdir(MedNIST_Hand_Dir)]

    # define real data transforms
    train_transforms = Compose(
        [
            LoadPNG(image_only=True),
            AddChannel(),
            ScaleIntensity(),
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            ToTensor(),
        ]
    )

    # create dataset and dataloader
    train_ds = CacheDataset(hands, train_transforms)
    batch_size = 300
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10)

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

    key_train_metric = None  # TODO: Make FID Monai Metric to evaluate generator performance

    # Create Adversarial Trainer
    disc_train_steps = 5
    num_epochs = 50

    trainer = AdversarialTrainer(
        device,
        num_epochs,
        train_loader,
        gen_net,
        gen_opt,
        generator_loss,
        disc_net,
        disc_opt,
        discriminator_loss,
        latent_shape=latent_size,
        d_train_steps=disc_train_steps,
        key_train_metric=key_train_metric,
        train_handlers=handlers,
    )

    # Run Training
    trainer.run()

    # The separate loss values for the generator and discriminator can be graphed together. These should reach an equilibrium as the generator's ability to fool the discriminator balances with that networks ability to discriminate accurately between real and fake images.
    # TODO: replicate graph printout with trainers output
    # plt.figure(figsize=(12, 5))
    # plt.semilogy(*zip(*gen_step_loss), label="Generator Loss")
    # plt.semilogy(*zip(*disc_step_loss), label="Discriminator Loss")
    # plt.grid(True, "both", "both")
    # plt.legend()
    # plt.savefig("GAN_LOSS_PLOT.png")

    # Save a few randomly generated images. Hopefully most images will have four fingers and a thumb as expected
    # (assuming polydactyl examples were not present in large numbers in the dataset).

    print("Saving trained generator data samples.")
    # TODO: turn into gan_save_fakes handler
    test_img_count = 10
    test_latent = torch.randn(test_img_count, latent_size).to(device)
    test_images = gen_net(test_latent)
    save_images(run_dir, "final", test_images)


if __name__ == "__main__":
    main()
