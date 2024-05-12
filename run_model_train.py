import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, sampler, TensorDataset
from torch.utils.data import sampler

import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

import matplotlib.pyplot as plt

import math
import numpy as np
from PIL import Image

from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter

from pathlib import Path

from time import time

import models as models

from torch.autograd import Variable
from tqdm import tqdm

import os

#-------------------------------------------#

UPSCALE_FACTOR = 2
NUM_EPOCHS = 1

base_folder = '/Volumes/DataDrive/clim_model_runs'
run_number = 1


#-------------------------------------------#
USE_GPU = False

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 1

print('using device:', device)

train_directory = '/Volumes/DataDrive/clim_tif/train'
test_directory = '/Volumes/DataDrive/clim_tif/test'
val_directory = '/Volumes/DataDrive/clim_tif/val'

##########################################################################################
#                                  BATCH_SIZE PARAMETER
BATCH_SIZE = 1
##########################################################################################


sr_train = models.SR_Dataset(train_directory)
loader_train = DataLoader(sr_train, batch_size=BATCH_SIZE)


sr_val = models.SR_Dataset(val_directory)
loader_val = DataLoader(sr_val, batch_size=BATCH_SIZE)

sr_test = models.SR_Dataset(test_directory)
loader_test = DataLoader(sr_test, batch_size=BATCH_SIZE)


#--------------------------------------------------#
netG = models.Generator(1,UPSCALE_FACTOR)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = models.Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

mseLoss = nn.MSELoss()

#---------------------------------------------------#

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(loader_train)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    netG.train()
    netD.train()
    for data, target in train_bar:
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        ##################################  
        # (1) Update D nework 
        # we want real_out to be close 1, and fake_out to be close 0 
        # maximize D(x) - D(G(z)) + [1] x: real_out D(G(z)): fake_out 
        ##################################
        real_img = Variable(target)
        #print(f"Real Image Shape: {real_img.shape}")
        z = Variable(data)
        #print(f"z Variable Shape: {z.shape}")
        fake_img = netG(z)
        #print(f"Fake Image Shape: {fake_img.shape}")

        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)


        ###################################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###################################
        netG.zero_grad()
        g_loss = mseLoss(fake_out, real_img)
        g_loss.backward()

        optimizerD.step()
        optimizerG.step()

        # loss for current batch before optimization 
        running_results['g_loss'] += g_loss.item() * batch_size
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.item() * batch_size
        running_results['g_score'] += fake_out.item() * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'], 
            running_results['g_loss'] / running_results['batch_sizes'], 
            running_results['d_score'] / running_results['batch_sizes'], 
            running_results['g_score'] / running_results['batch_sizes']
        ))

    netG.eval()


modelg_file = os.path.join(base_folder,f'g_model_run_{run_number}.pth')
modeld_file = os.path.join(base_folder,f'd_model_run_{run_number}.pth')

torch.save(netG.state_dict(), modelg_file)
torch.save(netD.state_dict(), modeld_file)