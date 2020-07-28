# Generative Adversarial Networks with MedNIST Dataset
# 
# This notebook illustrates the use of MONAI for training a network to generate images from a random input tensor. A simple GAN is employed to do with a separate Generator and Discriminator networks. 
#
# ### Get the dataset
# 
# The MedNIST dataset was gathered from several sets from [TCIA](https://wiki.cancerimagingarchive.net/display/Public/Data+Usage+Policies+and+Restrictions), [the RSNA Bone Age Challenge](http://rsnachallenges.cloudapp.net/competitions/4), and [the NIH Chest X-ray dataset](https://cloud.google.com/healthcare/docs/resources/public-datasets/nih-chest).
# 
# The dataset is kindly made available by [Dr. Bradley J. Erickson M.D., Ph.D.](https://www.mayo.edu/research/labs/radiology-informatics/overview) (Department of Radiology, Mayo Clinic)
# under the Creative Commons [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).
# If you use the MedNIST dataset, please acknowledge the source, e.g.
# 
# https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/mednist_tutorial.ipynb.
# 


import os
import sys

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

import monai
from monai.transforms import Transform, Compose, AddChannel, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom, ToNumpy
from monai.networks.nets import Generator, Discriminator
from monai.networks import normal_init
from adversarial_trainer import AdversarialTrainer
from monai.utils import progress_bar

np.random.seed(0)

MedNIST_Hand_Dir = '/shahinaaz/nvdata/MedNIST/Hand'

disc_train_interval = 1
disc_train_steps = 5
batch_size = 300
latent_size = 64
num_epochs = 50
real_label = 1
gen_label = 0
learning_rate = 2e-4
betas = (0.5, 0.999)

device = torch.device('cuda:0')

# load data

hands = [os.path.join(MedNIST_Hand_Dir, filename) for filename in os.listdir(MedNIST_Hand_Dir)]

class LoadJpeg(Transform):
    def __call__(self, data):
        return Image.open(data)
    

train_transforms = Compose([
    LoadJpeg(),
    ToNumpy(),
    AddChannel(),
    ScaleIntensity(),
    RandRotate(range_x=15, prob=0.5, keep_size=True),
    RandFlip(spatial_axis=0, prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ToTensor()
])

train_ds = monai.data.CacheDataset(hands, train_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10)

# define networks

disc_net=Discriminator(
    in_shape=(1, 64, 64),
    channels=(8, 16, 32, 64, 1),
    strides=(2,  2,  2,  2, 1), 
    num_res_units=1,
    kernel_size=5
).to(device)

gen_net = Generator(
    latent_shape=latent_size,
    start_shape=(64, 8, 8),
    channels=[32, 16, 8, 1],
    strides=[2, 2, 2, 1]
)

# initialize both networks 
disc_net.apply(normal_init)
gen_net.apply(normal_init)

# input images are scaled to [0,1] so enforce the same of generated outputs 
gen_net.conv.add_module('activation', torch.nn.Sigmoid()) 
gen_net = gen_net.to(device)


# Loss and Optimizers

disc_loss = torch.nn.BCELoss()
gen_loss = torch.nn.BCELoss()

disc_opt = torch.optim.Adam(disc_net.parameters(), learning_rate, betas=betas)
gen_opt = torch.optim.Adam(gen_net.parameters(), learning_rate, betas=betas)
 

def discriminator_loss(gen_images, real_images):
    """
    The discriminator loss if calculated by comparing its
    prediction for real and generated images.

    """
    real = real_images.new_full((real_images.shape[0], 1), real_label)
    gen = gen_images.new_full((gen_images.shape[0], 1), gen_label)

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


# We now train by iterating over the dataset for a number of epochs. At certain after the generator training stage for each batch, the discriminator is trained for a number of steps on the same real and generated images.

## TRAINING SCRIPT

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = [(0, 0)]
gen_step_loss = []
disc_step_loss = []
step = 0

trainer = AdversarialTrainer(device, num_epochs, train_loader, 
                             gen_net, gen_opt, gen_loss, 
                             disc_net, disc_opt, disc_loss
                             )

trainer.run()


'''
for epoch in range(num_epochs):
    gen_net.train()
    disc_net.train()
    epoch_loss = 0
    
    for i, batch_data in enumerate(train_loader):
        progress_bar(i, len(train_loader), f"epoch {epoch + 1}, avg loss: {epoch_loss_values[-1][1]:.4f}")
        real_images = batch_data.to(device)
        latent = torch.randn(real_images.shape[0], latent_size).to(device)
        
        gen_opt.zero_grad()
        gen_images = gen_net(latent)
        loss = generator_loss(gen_images)
        loss.backward()
        gen_opt.step()
        epoch_loss += loss.item()
        
        gen_step_loss.append((step,loss.item()))
        
        if step % disc_train_interval == 0:
            disc_total_loss = 0
            
            for _ in range(disc_train_steps): # 5
                disc_opt.zero_grad()
                dloss = discriminator_loss(gen_images, real_images)
                dloss.backward()
                disc_opt.step()
                disc_total_loss += dloss.item()
                
            disc_step_loss.append((step, disc_total_loss / disc_train_steps))
            
        step += 1
        # print(step)
        
    epoch_loss /= step
    # print(epoch_loss)
    epoch_loss_values.append((step, epoch_loss))
'''    

# The separate loss values for the generator and discriminator can be graphed together. These should reach an equilibrium as the generator's ability to fool the discriminator balances with that networks ability to discriminate accurately between real and fake images.


plt.figure(figsize=(12, 5))
plt.semilogy(*zip(*gen_step_loss), label='Generator Loss')
plt.semilogy(*zip(*disc_step_loss), label='Discriminator Loss')
plt.grid(True, 'both', 'both')
plt.legend()
plt.savefig('GAN_LOSS_PLOT.png')


# Finally we show a few randomly generated images. Hopefully most images will have four fingers and a thumb as expected (assuming polydactyl examples were not present in large numbers in the dataset). This demonstrative notebook doesn't train the networks for long, training beyond the default 50 epochs should improve results.

test_size = 10
test_latent = torch.randn(test_size, latent_size).to(device)

test_images = gen_net(test_latent)

for i in range(test_size):
    plt.figure()
    plt.imshow(test_images[i, 0].cpu().data.numpy(), cmap='gray')
    plt.savefig('GAN_OUTPUT_%d' % i)
