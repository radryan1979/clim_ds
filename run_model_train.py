import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, sampler, TensorDataset
from torch.utils.data import sampler
import torchvision.utils as utils

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

from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform

from pathlib import Path

from time import time

import models as models

from torch.autograd import Variable
from tqdm import tqdm

import os

import pandas as pd

import pytorch_ssim

# import warnings
# warnings.filterwarnings('ignore')

#-------------------------------------------#

UPSCALE_FACTOR = 2
NUM_EPOCHS = 2

base_folder = '/Volumes/DataDrive/clim_model_runs/'
out_path = '/Volumes/DataDrive/clim_model_runs/'
run_number = 8


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
print(netG)
netD = models.Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
print(netD)

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

mseLoss = nn.MSELoss()

#---------------------------------------------------#

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

for epoch in range(1, NUM_EPOCHS + 1):
    print(epoch)
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
        #z = Variable(data)
        #print(f"z Variable Shape: {z.shape}")
        fake_img = netG(data)
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
    
    with torch.no_grad():
        val_bar = tqdm(loader_val)
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_images = []
        for val_lr, val_hr in val_bar:
            batch_size = val_lr.size(0)
            valing_results['batch_sizes'] += batch_size
            lr = val_lr
            hr = val_hr
            if torch.cuda.is_available():
                lr = lr.cuda()
                hr = hr.cuda()
            sr = netG(lr)

            batch_mse = ((sr - hr) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(sr, hr).item()
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = 10 * np.log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
            val_bar.set_description(
                desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr'], valing_results['ssim']
                )
            )

            val_images.extend(
                [display_transform()(hr.data.cpu().squeeze(0)), 
                    display_transform()(sr.data.cpu().squeeze(0))]
            )
        val_images = torch.stack(val_images)
        val_images = torch.chunk(val_images, val_images.size(0) // 20)
        val_save_bar = tqdm(val_images, desc='[saving training results]')
        index = 1
        for image in val_save_bar:
            print(f'Image index: {index}')
            print(image.shape)
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
            index += 1

    # save model parameters
    gmodel = os.path.join(out_path, f'epochs/netG_epoch_{run_number}_{epoch}.pth')
    dmodel = os.path.join(out_path, f'epochs/netD_epoch_{run_number}_{epoch}.pth')
    
    torch.save(netG.state_dict(), gmodel)
    torch.save(netD.state_dict(), dmodel)
    # save loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    stat_path = os.path.join(out_path,"statistics/")
    print(stat_path)
    data_frame = pd.DataFrame(
        data = {'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'], 
                'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim'], }, 
                index=range(1, epoch+1)
    )
    stat_file = f'stats_epoch{epoch}_run{run_number}.csv'
    data_frame.to_csv(os.path.join(stat_path, stat_file), index_label='Epoch')