
import os, pickle
import numpy as np
import pandas as pd
from support_funs_GI import stopifnot, torch2array, array_plot, sigmoid
from time import time
from support_funs_GI import intax3
import torch
from unet_model import UNet

from helper_torch import CellCounterDataset, img2tensor, randomRotate, randomFlip
from torchvision import transforms
from torch.utils import data


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
device = torch.device('cuda' if use_cuda else 'cpu')

# Get the mean number of cells
pfac = 9  # Number of cells have been multiplied by 9 (basically)
mu_pixels = np.mean([intax3(di_img_point[z]['lbls']).mean() for z in di_img_point])
mu_cells = np.mean([di_img_point[z]['pts'].shape[0] for z in di_img_point])
print('Error: %0.1f' % (mu_pixels*501**2/pfac - mu_cells))
b0 = np.log(mu_pixels / (1 - mu_pixels))

# Load the model
torch.manual_seed(1234)
mdl = UNet(n_channels=3, n_classes=1, bilinear=False)
mdl.to(device)
with torch.no_grad():
    mdl.outc.conv.bias.fill_(b0)
# Check CUDA status for model
print('Are network parameters cuda?: %s' %
      all([z.is_cuda for z in mdl.parameters()]))

# Binary loss
criterion = torch.nn.BCEWithLogitsLoss()
# Optimizer
optimizer = torch.optim.Adagrad(params=mdl.parameters(), lr=1e-4)


# tnow = time()
# # Loop through images and make sure they average number of cells equal
# mat = np.zeros([len(ids_tissue), 2])
# for ii, idt in enumerate(ids_tissue):
#     print('ID-tissue %s (%i of %i)' % (idt, ii + 1, len(ids_tissue)))
#     tens = array2torch(di_img_point[idt]['img'], device)
#     arr = torch2array(tens)
#     with torch.no_grad():
#         logits = mdl(tens)
#         ncl = logits.cpu().mean().numpy()+0
#         nc = torch.sigmoid(logits).cpu().sum().numpy()+0
#         mat[ii] = [nc, ncl]
# # print(mat.mean(axis=0)); print(mu_pixels*501**2); print(mu_cells); print(b0)
# print('Script took %i seconds' % (time() - tnow))

################################
## --- (2) BEGIN TRAINING --- ##

# Select instances for training/validation
idt_val = list(np.random.choice(ids_tissue,3))
idt_train = list(np.setdiff1d(ids_tissue, idt_val))
print('%i training samples\n%i validation samples' %
      (len(idt_train), len(idt_val)))

# Create datasetloader class
train_params = {'batch_size': 2,'shuffle': True}
val_params = {'batch_size': len(idt_val),'shuffle': False}

train_transform = transforms.Compose([randomRotate(tol=1e-4), randomFlip(), img2tensor(device)])
train_data = CellCounterDataset(di=di_img_point, ids=idt_train, transform=train_transform)
train_gen = data.DataLoader(dataset=train_data,**train_params)
# ids_batch, lbls_batch, imgs_batch = next(iter(train_gen))
val_transform = transforms.Compose([img2tensor(device)])
val_data = CellCounterDataset(di=di_img_point, ids=idt_val, transform=val_transform)
val_gen = data.DataLoader(dataset=val_data,**val_params)

tnow = time()
nepochs = 500
mat_loss = np.zeros([nepochs, 2])
for ee in range(nepochs):
    print('--------- EPOCH %i of %i ----------' % (ee+1, nepochs))
    ii = 0
    np.random.seed(ee)
    holder_loss = []
    for ids_batch, lbls_batch, imgs_batch in train_gen:
        ii += 1
        ids_batch = list(ids_batch)
        print('-- batch %i of %i: %s --' % (ii, len(train_gen), ', '.join(ids_batch)))
        # --- Forward pass --- #
        optimizer.zero_grad()
        logits = mdl(imgs_batch)
        print(logits.mean())
        assert logits.shape == lbls_batch.shape
        loss = criterion(input=logits,target=lbls_batch)
        # --- Backward pass --- #
        loss.backward()
        # --- Gradient step --- #
        optimizer.step()
        # Empty cache
        torch.cuda.empty_cache()
        # Get performance
        ii_loss = loss.cpu().detach().numpy()+0
        # PREDICTED VERSUS ACTUAL GOES HERE
        holder_loss.append(ii_loss)
        print('Loss: %0.4f' % (ii_loss))
    ee_loss = np.mean(holder_loss)
    # Evaluate model on validation data
    with torch.no_grad():
        ids_batch, lbls_batch, imgs_batch = next(iter(val_gen))
        logits = mdl.eval()(imgs_batch)
        vv_loss = criterion(input=logits,target=lbls_batch).cpu().detach().numpy()+0
        # Empty cache
        torch.cuda.empty_cache()
    print('Validation loss: %0.4f' % vv_loss)
    mat_loss[ee] = [ee_loss, vv_loss]
    print('Epoch took %i seconds' % int(time() - tnow))
    tnow = time()
    # Save plots
    if ee+1 == nepochs:
        # Training
        ids_batch, lbls_batch, imgs_batch = next(iter(train_gen))
        logits = mdl.eval()(imgs_batch)
        array_plot(torch2array(imgs_batch), pts=sigmoid(torch2array(logits[:, 0, :, :])),
                   path=dir_figures, fn='train_samples_' + str(ee + 1) + '.png')
        del ids_batch, lbls_batch, imgs_batch, logits
        # Validation
        ids_batch, lbls_batch, imgs_batch = next(iter(val_gen))
        logits = mdl.eval()(imgs_batch)
        array_plot(torch2array(imgs_batch), pts=sigmoid(torch2array(logits[:, 0, :, :])),
                   path=dir_figures, fn='val_samples_' + str(ee+1) + '.png')
        del ids_batch, lbls_batch, imgs_batch, logits

print(mat_loss)








