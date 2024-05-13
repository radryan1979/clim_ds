import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import numpy as np
import models2 as models
import loss as loss

from tqdm import tqdm
import os
import pandas as pd
import pytorch_ssim

# import warnings
# warnings.filterwarnings('ignore')

#-------------------------------------------#

UPSCALE_FACTOR = 2
NUM_EPOCHS = 25

base_folder = '/Volumes/DataDrive/clim_model_runs/'
out_path = '/Volumes/DataDrive/clim_model_runs/'
run_number = 11


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
BATCH_SIZE = 2
##########################################################################################


sr_train = models.SR_Dataset(train_directory)
loader_train = DataLoader(sr_train, batch_size=BATCH_SIZE)


sr_val = models.SR_Dataset(val_directory)
loader_val = DataLoader(sr_val, batch_size=BATCH_SIZE)

sr_test = models.SR_Dataset(test_directory)
loader_test = DataLoader(sr_test, batch_size=BATCH_SIZE)


#--------------------------------------------------#
netG = models.Generator(UPSCALE_FACTOR)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
print(netG)
netD = models.Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
print(netD)

# these explode before the first epoch is even done
# optimizerG = optim.Adam(netG.parameters(), lr=1e-5, betas=(0.5, 0.9))
# optimizerD = optim.Adam(netD.parameters(), lr=1e-5, betas=(0.5, 0.9))

optimizerG = optim.SGD(netG.parameters(), lr=1e-5)
optimizerD = optim.SGD(netD.parameters(), lr=1e-5)

mseLoss = nn.MSELoss()

#---------------------------------------------------#

results = {
        'd_loss_real':[],
        'd_loss_fake':[],
        'd_loss_total':[],
        'g_loss_content':[],
        'g_loss_adv':[],
        'g_loss_total':[],
        'psnr':[],
        'ssim':[]
    }

for epoch in range(1, NUM_EPOCHS + 1):
    print(epoch)
    train_bar = tqdm(loader_train)
    # running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    running_results = {
        'batch_sizes':0,
        'd_loss_real':0,
        'd_loss_fake':0,
        'd_loss_total':0,
        'g_loss_content':0,
        'g_loss_adv':0,
        'g_loss_total':0
    }

    netG.train()
    netD.train()
    
    G_content = np.zeros(len(loader_train)*NUM_EPOCHS+1)
    G_advers = np.zeros(len(loader_train)*NUM_EPOCHS+1)
    D_real_L = np.zeros(len(loader_train)*NUM_EPOCHS+1)
    D_fake_L = np.zeros(len(loader_train)*NUM_EPOCHS+1)
    
    iter_count = 0
    for data, target in train_bar:
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        ##################################  
        # (1) Update D nework 
        # we want real_out to be close 1, and fake_out to be close 0 
        # maximize D(x) - D(G(z)) + [1] x: real_out D(G(z)): fake_out 
        ##################################
        real_img = target
        logits_real = netD(real_img)
        fake_img = netG(data)
        logits_fake = netD(fake_img)
        
        # Update for the discriminator
        d_loss, D_real_L[iter_count], D_fake_L[iter_count] = loss.discriminator_loss(logits_real, logits_fake)
        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()

        fake_images = netG(data)
        logits_fake = netD(fake_images)
        gen_logits_fake = netD(fake_images)
        weight_param = 1e-1 # Weighting put on adversarial loss
        g_error, G_content[iter_count], G_advers[iter_count] = loss.generator_loss(fake_images, real_img, gen_logits_fake, weight_param=weight_param)
        optimizerG.zero_grad()
        g_error.backward()
        optimizerG.step()

        # loss for current batch before optimization 
        running_results['d_loss_real'] += D_real_L[iter_count]
        running_results['d_loss_fake'] += D_fake_L[iter_count]
        running_results['d_loss_total'] += d_loss
        running_results['g_loss_content'] += G_content[iter_count]
        running_results['g_loss_adv'] += G_advers[iter_count]
        running_results['g_loss_total'] += g_error



        train_bar.set_description(desc='[%d/%d] D Loss Real: %.4f D Loss Fake: %.4f D Loss Total: %.4f G Loss Content: %.4f G Loss Adv: %.4f G Loss Total: %.4f' % (
            epoch, NUM_EPOCHS,
            running_results['d_loss_real'] / running_results['batch_sizes'],
            running_results['d_loss_fake'] / running_results['batch_sizes'],
            running_results['d_loss_total'] / running_results['batch_sizes'],
            running_results['g_loss_content'] / running_results['batch_sizes'],
            running_results['g_loss_adv'] / running_results['batch_sizes'],
            running_results['g_loss_total'] / running_results['batch_sizes']
        ))
        
        iter_count += 1

    # put the model in eval model
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
                 [   lr.data.cpu().squeeze(0),
                     hr.data.cpu().squeeze(0), 
                     sr.data.cpu().squeeze(0)]
             )
            
        val_images = torch.stack(val_images)
        val_images = torch.chunk(val_images, val_images.size(0) // 20)
        val_save_bar = tqdm(val_images, desc='[saving training results]')
        # index = 1
        # print(f'Length of val_save_bar: {0}',len(val_save_bar))
        # for image in val_save_bar:
        #     #print(f'Image index: {index}')
        #     #print(image[0].shape)
        #     #print(image[1].shape)
        #     #print(image[3].shape)
        #     plt.subplot(131)
        #     plt.imshow(image[0][1,0,:,:])
        #     plt.title('Low Res Image')
        #     plt.subplot(132)
        #     plt.imshow(image[1][1,0,:,:])
        #     plt.title('High Res Image')
        #     plt.subplot(133)
        #     plt.imshow(image[2][1,0,:,:])
        #     plt.title('SR Image')
        #     saveloc = os.path.join(out_path,f'epoch_{epoch}_index_{index}.png')
        #     plt.savefig(saveloc)
        #     index += 1

    # save model parameters
    gmodel = os.path.join(out_path, f'epochs/netG_run_{run_number}_epoch_{epoch}.pth')
    dmodel = os.path.join(out_path, f'epochs/netD_run_{run_number}_epoch_{epoch}.pth')
    
    torch.save(netG.state_dict(), gmodel)
    torch.save(netD.state_dict(), dmodel)
    
    
    # save loss\scores\psnr\ssim
    
    results['d_loss_real'].append(running_results['d_loss_real'] / running_results['batch_sizes'])
    results['d_loss_fake'].append(running_results['d_loss_fake'] / running_results['batch_sizes'])
    results['d_loss_total'].append(running_results['d_loss_total'] / running_results['batch_sizes'])
    results['g_loss_content'].append(running_results['g_loss_content'] / running_results['batch_sizes'])
    results['g_loss_adv'].append(running_results['g_loss_adv'] / running_results['batch_sizes'])
    results['g_loss_total'].append(running_results['g_loss_total'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    stat_path = os.path.join(out_path,"statistics/")
    print(stat_path)
    data_frame = pd.DataFrame(
            data = {
                'd_loss_real':results['d_loss_real'],
                'd_loss_fake':results['d_loss_fake'],
                #'d_loss_total':results['d_loss_total'],
                'g_loss_content':results['g_loss_content'],
                'g_loss_adv':results['g_loss_adv'],
                #'g_loss_total':results['g_loss_total']
                'PSNR': results['psnr'], 
                'SSIM': results['ssim']
                }, 
            index=range(1, epoch+1)
    )
    stat_file = f'stats_epoch{epoch}_run{run_number}.csv'
    data_frame.to_csv(os.path.join(stat_path, stat_file), index_label='Epoch')