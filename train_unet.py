import os, pickle
import numpy as np
import pandas as pd
from support_funs_GI import stopifnot, array2torch, torch2array, array_plot
import torch
from unet_model import UNet

######################################
## --- (1) PREP DATA AND MODELS --- ##

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_output, 'figures')
# Load data
di_img_point = pickle.load(open(os.path.join(dir_output, 'di_img_point.pickle'), 'rb'))
ids_tissue = list(di_img_point.keys())
print(np.round(di_img_point[ids_tissue[0]]['lbls'].sum() / 9).astype(int))
print(di_img_point[ids_tissue[0]]['pts'].shape[0])

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device("cuda" if use_cuda else "cpu")

# Get the mean number of cells
mu_pixels = np.mean([di_img_point[z]['lbls'].mean() for z in di_img_point])
mu_cells = np.mean([di_img_point[z]['pts'].shape[0] for z in di_img_point])
b0 = np.log(mu_pixels / (1 - mu_pixels))
#### MAKE SURE THE INTECEPT MAKES SENSE !!! ######




# Load the model
torch.manual_seed(1234)
mdl = UNet(n_channels=3, n_classes=1, bilinear=False)
mdl.to(device)
with torch.no_grad():
    mdl.outc.conv.bias.fill_(b0)
# Check CUDA status for model
print('Are network parameters cuda?: %s' %
      all([z.is_cuda for z in mdl.parameters()]))

# Loop through images and make sure they average number of cells equal
holder = []
for ii, idt in enumerate(ids_tissue):
    print('ID-tissue %s (%i of %i)' % (idt, ii + 1, len(ids_tissue)))
    tens = array2torch(di_img_point[idt]['img'], device)
    arr = torch2array(tens)
    with torch.no_grad():
        logits = mdl(tens)
        ncl = logits.cpu().mean().numpy()+0
        # print(ncl)
        nc = torch.sigmoid(logits).cpu().sum().numpy()+0
        holder.append(ncl)
print(np.mean(holder))
print(b0)
print(mu_cells)

# array_plot(arr=arr, path=dir_figures, cmap='gray')  #np.apply_over_axes(np.mean,arr,2)[:,:,0]/255
# # Test image
# img = di_img_point[ids_tissue[0]]['img'].copy()
# print(logits.is_cuda)
# print(logits.mean())

################################
## --- (2) BEGIN TRAINING --- ##
