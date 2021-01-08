"""
1) CALCULATE DENSITY ON ENTIRE IMAGE
2) SAVES IN THE ~/OUTPUT/FIGURES/INFERENCE FOLDER
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-rid', '--ridx', type=int, help='Patient number to pick (0-188)', default=0)
parser.add_argument('-k', '--kk', type=int, help='The max box size for the convolution on entire image', default=1000)
args = parser.parse_args()
ridx = args.ridx
kk = args.kk
#ridx, kk = 0, 1000
print('ridx = %i, k = %i' % (ridx, kk))

import gc
import os
import numpy as np
import pandas as pd
from funs_support import sigmoid, t2n, makeifnot, find_dir_cell
import torch
from PIL import Image
from funs_unet import find_bl_UNet

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device('cuda' if use_cuda else 'cpu')

dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
dir_snapshot = os.path.join(dir_checkpoint, 'snapshot')
lst_dir = [dir_output, dir_figures, dir_checkpoint, dir_snapshot]
assert all([os.path.exists(path) for path in lst_dir])
dir_inference = os.path.join(dir_figures, 'inference')
makeifnot(dir_inference)

# Get the dates from the snapshot folder
fns_snapshot = pd.Series(os.listdir(dir_snapshot))
dates_snapshot = pd.to_datetime(fns_snapshot.str.split('\\.|\\_', 5, True).iloc[:, 2:5].apply(lambda x: '-'.join(x), 1))
dates2 = pd.Series(dates_snapshot.sort_values(ascending=False).unique())
dnew = dates2[0].strftime('%Y_%m_%d')
print('The current date is: %s' % dnew)
# Make folder in inference with the newest date
dir_save = os.path.join(dir_inference, dnew)
makeifnot(dir_save)

###########################################
## --- (1) LOAD DATA AND UNET MODEL  --- ##

# How the "cells" are labelled in the snapshot folder
cells = ['Eosinophil', 'Inflammatory']
# Valid tissue types
tissues = ['Rectum', 'Ascending', 'Sigmoid', 'Transverse', 'Descending', 'Cecum']

# Initialize two models
fn_eosin_new, fn_inflam_new = tuple([os.path.join(dir_snapshot, 'mdl_' + cell + '_' + dnew + '.pt') for cell in cells])
mdl_eosin_new = find_bl_UNet(path=fn_eosin_new, device=device, batchnorm=True,
                             start=12, stop=24, step=4)
mdl_inflam_new = find_bl_UNet(path=fn_inflam_new, device=device, batchnorm=True,
                              start=12, stop=24, step=4)

holder = []
for idt in os.listdir(dir_cleaned):
    fold = os.path.join(dir_cleaned, idt)
    fns = os.listdir(fold)
    holder.append(pd.DataFrame({'idt':idt, 'fn':fns}))
dat_IDs = pd.concat(holder).reset_index(None,True)
tmp = dat_IDs.fn.str.replace('cleaned_|.png','').str.split('_',1,True)
assert all(dat_IDs.idt == tmp.iloc[:,0])
tmp = tmp.iloc[:,1].str.split('\\-',1,True).rename(columns={0:'tissue', 1:'version'})
tmp.version = np.where(tmp.version.isnull(),1,2)
dat_IDs = dat_IDs.assign(tissue = tmp.tissue, version=tmp.version)
dat_IDs = dat_IDs[dat_IDs.tissue.isin(tissues)].reset_index(None,True)
print('There are a total of %i rows' % dat_IDs.shape[0])


#########################################
## --- (2) COMPARE TO ENTIRE IMAGE --- ##

mdl_eosin_new.eval()
mdl_inflam_new.eval()
torch.cuda.empty_cache()

print('Making sure we can run through GPU')
with torch.no_grad():
    print(mdl_eosin_new(torch.rand(1, 3, kk, kk).to(device)).shape)
    print(mdl_inflam_new(torch.rand(1, 3, kk, kk).to(device)).shape)
    torch.cuda.empty_cache()

ii, rr = ridx, dat_IDs.loc[ridx]
print('Image %i of %i' % (ii + 1, dat_IDs.shape[0]))
idt, tissue, file = rr['idt'], rr['tissue'], rr['fn']
# Load the image
path = os.path.join(dir_cleaned, idt, file)
assert os.path.exists(path)
img = Image.open(path)
img = np.array(img.convert('RGB'))
height, width, channels = img.shape
print(img.shape)
# Loop over the image in convolutional chunks
nr, nd = int(np.round(width / kk)), int(np.round(height / kk))
print('Number of rows, number of cols: %i, %i' % (nr, nd))
phat_eosin = np.zeros([height, width])
phat_inflam = phat_eosin.copy()
for r in range(nr):
    for d in range(nd):
        print('Convolution: r=%i, d=%i' % (r,d))
        tmp_img = img[d * kk:(d * kk) + kk, r * kk: (r * kk) + kk].copy()
        tmp_img = np.stack([tmp_img], axis=3).transpose([3, 2, 0, 1])
        tmp_img = torch.tensor(tmp_img / 255, dtype=torch.float32).to(device)
        with torch.no_grad():
            tmp_phat_eosin = sigmoid(t2n(mdl_eosin_new(tmp_img)))
            torch.cuda.empty_cache()
            tmp_phat_inflam = sigmoid(t2n(mdl_inflam_new(tmp_img)))
            torch.cuda.empty_cache()
        phat_eosin[d * kk:(d * kk) + kk, r * kk: (r * kk) + kk] = tmp_phat_eosin
        phat_inflam[d * kk:(d * kk) + kk, r * kk: (r * kk) + kk] = tmp_phat_inflam
        print('Num eosin: %0.1f, num inflam: %0.1f' % (phat_eosin.sum(), phat_inflam.sum()))

# # Test figure
# from funs_support import val_plt
# val_plt(img, np.dstack([phat_inflam]), np.dstack([phat_inflam]), lbls=['inflam'], path=dir_save, thresh=[1e-3], fn='phat_inflam.png')
num_eosin = phat_eosin.sum()
num_inflam = phat_inflam.sum()
ratio = num_eosin / num_inflam
print('Ratio for patient %s is: %0.3f%%' % (idt, ratio*100))
gc.collect()
del img, phat_eosin, phat_inflam, tmp_img, tmp_phat_eosin, tmp_phat_inflam
df_slice = pd.DataFrame(rr).T.assign(eosin=num_eosin, inflam=num_inflam)
fn = os.path.join(dir_save, 'df_slice_' + str(ridx) + '.csv')
df_slice.to_csv(fn, index=False)
