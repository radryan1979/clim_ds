
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, sampler, TensorDataset
from torch.utils.data import sampler
from torch.autograd import Variable
from tqdm import tqdm

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

class SR_Dataset(data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        self.img_list = list(Path(img_dir).glob('*.tif'))
        # import required module

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        hr_image = Image.open(img_path)
        high_res = np.array(hr_image)
        
        # get the input LR image from output HR image by blurring, cropping, then interpolating
        h1,w1 = high_res.shape
        blurred = np.zeros_like(high_res)
        blurred = gaussian_filter(high_res, sigma = (0.55, 0.55))
        half_res = blurred[::2, ::2]
        
        # Code involved in interpolating the blurred image back up to high res resolution
        h2,w2 = half_res.shape
        x = np.arange(h2)
        y = np.arange(w2)
        xnew = np.arange(0, h2, h2/h1)
        ynew = np.arange(0, w2, w2/w1)
        low_res = np.zeros_like(high_res)
        f = RectBivariateSpline(x, y, half_res[:, :])
        low_res[:, :] = f(xnew, ynew)


        low_res_t = torch.nan_to_num(torch.from_numpy(low_res).float())
        high_res_t = torch.nan_to_num(torch.from_numpy(high_res).float())
        
        return low_res_t.unsqueeze(0), high_res_t.unsqueeze(0)

# Convenience 
class Flatten(nn.Module):
    def forward(self, x):
        #N, C, H, W = x.size() # read in N, C, H, W
        return x.view(1, -1)  # "flatten" the C * H * W values into a single vector per image

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        linear_in = int(512*np.ceil(64/16)*np.ceil(64/16))
        
        self.layers = nn.Sequential( 
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            #nn.BatchNorm2d(64),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3),
            #nn.BatchNorm2d(128),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            #nn.BatchNorm2d(128),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3),
            #nn.BatchNorm2d(256),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            #nn.BatchNorm2d(256),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3),
            #nn.BatchNorm2d(512),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2),
            #nn.BatchNorm2d(512),
            #nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            
            Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
            
        )
        
    def forward(self, x):
        return self.layers(x)

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm2d(64),
            nn.InstanceNorm2d(64),
            nn.PReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            #nn.BatchNorm2d(64)
        )

    def forward(self, x):
        return x + self.layers(x)

class UpscaleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 256, 3, stride=1, padding=0),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )
    
    def forward(self, x):
        return self.layers(x)

class Generator(nn.Module):
    def __init__(self, num_channels, num_res_blocks=16, scale_factor=2):
#         upsample_block_num = int(math.log(scale_factor, 2))

        super().__init__()
        # Store the number of residual blocks, we need this number in the forward() function
        self.num_res_blocks = num_res_blocks
        
        self.initial_conv = nn.Sequential(
            nn.ReplicationPad2d(4),
            nn.Conv2d(num_channels, 64, kernel_size=9, stride=1, padding=0),
            nn.PReLU()
        )
        
        # List of residual blocks
        self.resBlocks = nn.ModuleList([ResidualBlock(64) for i in range(self.num_res_blocks)])

        self.post_resid_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            #nn.BatchNorm2d(64)
        )
    
        self.conv_prelu = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.PReLU()
        )
    
        self.final_conv = nn.Sequential(
            nn.ReplicationPad2d(4),
            nn.Conv2d(64, 1, 9, stride=1, padding=0)
        )


    def forward(self, x):
        initial_conv_out = self.initial_conv(x)
                
        res_block_out = self.resBlocks[0](initial_conv_out)
        for i in range(1, self.num_res_blocks):
            res_block_out = self.resBlocks[i](res_block_out)

        post_resid_conv_out = self.post_resid_conv(res_block_out) + initial_conv_out
        conv_prelu_out = self.conv_prelu(post_resid_conv_out)
        final_out = self.final_conv(conv_prelu_out)


        return F.tanh(final_out) 


