# Adapted models from Vandal implementation
# these run and work better in training
# output of the generator isn't working right
# data normalization or layer problem

import math
import torch
from torch import nn

import torch.optim as optim
import torch.utils.data as data

import numpy as np
from PIL import Image

from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter

from pathlib import Path

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

class Generator(nn.Module):
    def __init__(self, scale_factor) -> None:
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4), 
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBlock(64, 1) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2), 

            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(512, 1024, kernel_size=1), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual
    

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (up_scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x