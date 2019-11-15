#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
thanks to `Nathan Inkawhich <https://github.com/inkawhich>` for original
DCGAN implementation
"""

from __future__ import print_function
import sys
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)

print('\n\n >>>>>> DCGAN >>>>>>')
######################################################################
# Inputs

dataroot    = "/u/xl/tjlane/cryoem/dynanet/particle_simulations/ortho/kdef"
workers     = 16
batch_size  = 128
image_size  = 64
nc          = 1
ngf         = 64
ndf         = 64
lr          = 0.0002
beta1       = 0.5
noise_level = 0.1
ngpu        = 8

nz         = int(sys.argv[-3])
ortho_beta = float(sys.argv[-2])
num_epochs = int(sys.argv[-1])

print(' ------ PARAMETERS --------')
print(' > nz     :  %d' % nz)
print(' > obeta  :  %.2f' % ortho_beta)
print(' > epochs :  %d' % num_epochs)
print(' > rseed  :  %d' % manualSeed)
print('')


# decide on a place to put results
bas_dir    = '/u/xl/tjlane/cryoem/dynanet/particle_simulations/ortho/kdef/models'
res_dir    = 'nz%d_ortho%.2f_epoch%d' % (nz, ortho_beta, num_epochs)
out_dir    = os.path.join(bas_dir, res_dir)
if os.path.exists(out_dir):
    os.system('rm -r %s' % out_dir)
os.mkdir(out_dir)
print('Saving results to -->')
print('  %s' % out_dir)

# Data
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



# -----------------------------------------------------------------------------
# Create the generator
netG = Generator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)
print('\n--- GENERATOR ---')
print(netG)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
print('\n--- DISCRIMINATOR ---')
print(netD)

# loss functions
criterion = nn.BCELoss()


# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# -----------------------------------------------------------------------------
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("\n\nStarting Training Loop...\n")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)

        # soft, noisy labels
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label)
        label += torch.randn(*label.shape) * noise_level
        label = label.clamp(0,1).to(device)

        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errGC = criterion(output, label)
        # Calculate gradients for G
        errGC.backward()
        D_G_z2 = output.mean().item()

        # -- tjl addition
        #errGJ = ortho_beta * jg_loss(netG, noise, image_size*image_size, 
        #                             reduction='mean')
        #errJ.backward()
        errG = errGC #+ errGJ
        # ^^^^^^^^^^^^^^^

        # Update G
        optimizerG.step()
        
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1


######################################################################
# Save the models

params_dir = os.path.join(out_dir, 'params')
if not os.path.exists(params_dir):
    os.mkdir(params_dir)

print('saving final model parameters...')

torch.save(netD.state_dict(), os.path.join(params_dir, 'netD.pk'))
torch.save(netG.state_dict(), os.path.join(params_dir, 'netG.pk'))

torch.save(optimizerD.state_dict(), os.path.join(params_dir, 'optD.pk'))
torch.save(optimizerG.state_dict(), os.path.join(params_dir, 'optG.pk'))

######################################################################
# Results
# -------

# training curves
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig( os.path.join(out_dir, 'training_curves.png') )
print(' --> saving: training_curves.png')


# real vs fake

real_batch = next(iter(dataloader))

plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig( os.path.join(out_dir, 'real_v_fake.png') )
print(' --> saving: real_v_fake.png')


# look into latent space

latent_dir = os.path.join(out_dir, 'latent')
if not os.path.exists(latent_dir):
    os.mkdir(latent_dir)

start  = -2
end    =  2
steps  = 64
nz_max = nz

dz = torch.zeros(64, nz, 1, 1).to(device)
d = torch.linspace(start, end, steps=steps).to(device)

with torch.no_grad():
    for iz in range(nz_max):
        
        dz[:,iz,0,0] = d
        
        fake_dz = netG(dz).detach().cpu()
        grid = vutils.make_grid(fake_dz, padding=5, normalize=True)
        
        plt.figure(figsize=(12,12))
        plt.subplot(111)
        plt.axis("off")
        plt.title("z_%d" % iz)
        plt.imshow(np.transpose(grid, (1,2,0)))
        lfn = os.path.join(latent_dir, 'dz_%d.png' % iz)
        plt.savefig(lfn)
        print(' --> saving: %s' % lfn)
        plt.close()

        dz[:,iz,0,0] = 0.0

print('... done')


