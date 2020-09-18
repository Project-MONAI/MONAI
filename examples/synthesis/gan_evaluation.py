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
MONAI GAN Evaluation Example
    Generate fake images from trained generator file.

"""

import logging
import os
import sys
from glob import glob

import torch

import monai
from monai.data import png_writer
from monai.engines.utils import default_make_latent as make_latent
from monai.networks.nets import Generator
from monai.utils.misc import set_determinism


def save_generator_fakes(run_folder, g_output_tensor):
    for i, image in enumerate(g_output_tensor):
        filename = "gen-fake-%d.png" % i
        save_path = os.path.join(run_folder, filename)
        img_array = image[0].cpu().data.numpy()
        png_writer.write_png(img_array, save_path, scale=255)


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    set_determinism(12345)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load generator
    network_filepath = glob("./model_out/*.pth")[0]
    data = torch.load(network_filepath)
    latent_size = 64
    gen_net = Generator(
        latent_shape=latent_size, start_shape=(latent_size, 8, 8), channels=[32, 16, 8, 1], strides=[2, 2, 2, 1]
    )
    gen_net.conv.add_module("activation", torch.nn.Sigmoid())
    gen_net.load_state_dict(data["g_net"])
    gen_net = gen_net.to(device)

    # create fakes
    output_dir = "./generated_images"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    num_fakes = 10
    print("Generating %d fakes and saving in %s" % (num_fakes, output_dir))
    fake_latents = make_latent(num_fakes, latent_size).to(device)
    save_generator_fakes(output_dir, gen_net(fake_latents))


if __name__ == "__main__":
    main()
