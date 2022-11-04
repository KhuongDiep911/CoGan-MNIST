import argparse
import os
import numpy as np
import math
import scipy
import itertools

import mnistm

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension") #32 --> 28
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    # if isinstance(m, nn.Linear):
    #     nn.init.normal_(m.weight, 0.0, 0.02)
    # if isinstance(m, nn.BatchNorm2d):
    #     nn.init.normal_(m.weight, 1.0, 0.02)
    #     nn.init.constant_(m.bias, 0.0)
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) #m.weight.data --> m.weight
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class CoupledGenerators(nn.Module):
    def __init__(self):
        super(CoupledGenerators, self).__init__()

        self.init_size = opt.img_size // 4
        # self.fc = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))
        # self.latent_dim = latent_dim
        self.shared_conv1 = nn.Sequential(
            nn.ConvTranspose2d(opt.latent_dim, 1024, kernel_size=4, stride=1), #opt.latent_dim --> 128
            nn.BatchNorm2d(1024),# affine=False), 
            nn.PReLU()
        )
        self.shared_conv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),# affine=False), 
            nn.PReLU()
        )
        self.shared_conv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),# affine=False), 
            nn.PReLU()
        )
        self.shared_conv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),# affine=False), 
            nn.PReLU()
        )
        self.G1 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=6, stride=1, padding=1),
            nn.Tanh()
        )
        self.G2 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=6, stride=1, padding=1),
            nn.Tanh()
        )
    def forward(self, noise):
        
        # out = self.fc(noise)
        # print(out)
        # noise = noise.view(noise.shape[0], 128, self.init_size, self.init_size)
        # out = noise.view(noise.shape[0], opt.latent_dim, 1, 1)#self.init_size, self.init_size)
        out = noise.view(noise.shape[0], opt.latent_dim, 1, 1)
        x = self.shared_conv1(out)
        x = self.shared_conv2(x)
        x = self.shared_conv3(x)
        x = self.shared_conv4(x)
        img1 = self.G1(x)
        img2 = self.G2(x)
        return img1, img2

class CoupledDiscriminators(nn.Module):
    def __init__(self):
        super(CoupledDiscriminators, self).__init__()

        self.D1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2)
        )
        self.D2 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2)
        )
        self.shared_conv = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        # self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(800,500), 
            nn.PReLU(),
            nn.Linear(500,1), 
            nn.Sigmoid()
        )
        # self.fcD = nn.Sequential(
        #     nn.Linear(500,1), 
        #     nn.Sigmoid()
        # )
        # self.fcD2 = nn.Sequential(
        #     nn.Linear(500,1), 
        #     nn.Sigmoid()
        # )
    def forward(self, x_a, x_b):
        h0_a = self.D1(x_a)
        # print(h0_a.shape)
        h0_b = self.D2(x_b)
        # print(h0_b.shape)
        h0 = torch.cat((h0_a, h0_b), 0)
        # print(h0.shape)
        h1 = self.shared_conv(h0)
        # print(h1.shape[0])
        h1_f = h1.view(h1.shape[0], -1)#self.flatten(h1)
        # print(h1_f.shape)
        validity = self.fc(h1_f)
        # print(h2.shape)
        # validity = self.fcD(h2)
        # print(validity1.shape)
        validity1 = validity[:32]
        validity2 = validity[32:]
        # print(validity2.shape)
        return validity1, validity2 #h3.squeeze()
       


# Loss function
adversarial_loss = torch.nn.MSELoss()

# Initialize models
coupled_generators = CoupledGenerators()
coupled_discriminators = CoupledDiscriminators()

if cuda:
    coupled_generators.cuda()
    coupled_discriminators.cuda()

# Initialize weights
# print('ssssssss')
coupled_generators.apply(weights_init_normal)
coupled_discriminators.apply(weights_init_normal)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader1 = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

os.makedirs("../../data/mnistm", exist_ok=True)
dataloader2 = torch.utils.data.DataLoader(
    mnistm.MNISTM(
        "../../data/mnistm",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(coupled_generators.parameters(), lr=opt.lr)#, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(coupled_discriminators.parameters(), lr=opt.lr)#, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, ((imgs1, _), (imgs2, _)) in enumerate(zip(dataloader1, dataloader1)):

        batch_size = imgs1.shape[0]

        # Adversarial ground truths
        valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False) #batch_size*2
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        imgs1 = Variable(imgs1.type(Tensor).expand(imgs1.size(0), 1, opt.img_size, opt.img_size)) ###3-->1
        # print(imgs1.shape)
        # print(imgs2.shape)
        imgs2 = Variable(imgs2.type(Tensor))#.expand(imgs1.size(0), 1, opt.img_size, opt.img_size))
        # print(imgs2.shape)
        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        gen_imgs1, gen_imgs2 = coupled_generators(z)
        # print(gen_imgs1.shape)
        # Determine validity of generated images
        validity1, validity2 = coupled_discriminators(gen_imgs1, gen_imgs2)
        # validity = coupled_discriminators(gen_imgs1, gen_imgs2)
        # print(valid.shape)
        # print(validity1.shape)
        g_loss = (adversarial_loss(validity1, valid) + adversarial_loss(validity2, valid)) / 2
        # g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ----------------------
        #  Train Discriminators
        # ----------------------

        optimizer_D.zero_grad()
        # print(imgs1.shape)
        # print(gen_imgs1.shape)
        # Determine validity of real and generated images
        # print(imgs1.shape)
        # print(imgs2.shape)
        validity1_real, validity2_real = coupled_discriminators(imgs1, imgs2)
        
        validity1_fake, validity2_fake = coupled_discriminators(gen_imgs1.detach(), gen_imgs2.detach())
        # validity_real = coupled_discriminators(imgs1, imgs2)
        # validity_fake = coupled_discriminators(gen_imgs1.detach(), gen_imgs2.detach())


        d_loss = (
            adversarial_loss(validity1_real, valid)
            + adversarial_loss(validity1_fake, fake)
            + adversarial_loss(validity2_real, valid)
            + adversarial_loss(validity2_fake, fake)
        ) / 4
        # d_loss = (
        #     adversarial_loss(validity_real, valid)
        #     + adversarial_loss(validity_fake, fake)
        # ) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader1), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader1) + i
        if batches_done % opt.sample_interval == 0:
            gen_imgs = torch.cat((gen_imgs1.data, gen_imgs2.data), 0)
            save_image(gen_imgs, "images/%d.png" % batches_done, nrow=8, normalize=True)
