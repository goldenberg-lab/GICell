# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--save_model', dest='save_model', action='store_true', help='Save model as .pt file')
# parser.add_argument('--check_int', dest='check_int', action='store_true', help='Should forward pass be done to print intercept?')
# parser.add_argument('--check_model', dest='check_model', action='store_true', help='Stop model after one epoch')
# parser.add_argument('--is_eosin', dest='is_eosin', action='store_true', help='Eosinophil cell only')
# parser.add_argument('--is_inflam', dest='is_inflam', action='store_true', help='Eosinophil + neutrophil + plasma + lymphocyte')
# parser.set_defaults(is_eosin=False, is_inflam=False)
# parser.add_argument('--nepoch', type=int, default=110, help='Number of epochs')
# parser.add_argument('--batch', type=int, default=1, help='Batch size')
# parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
# parser.add_argument('--p', type=int, default=8, help='Number of initial params for NNet')
# parser.add_argument('--nfill', type=int, default=1, help='How many points to pad around pixel annotation point')
# args = parser.parse_args()
# save_model, check_int, check_model = args.save_model, args.check_int, args.check_model
# is_eosin, is_inflam = args.is_eosin, args.is_inflam
# nepoch, batch, lr, p, nfill = args.nepoch, args.batch, args.lr, args.p, args.nfill

# if save_model:
#     print('!!! WARNING --- MODEL WILL BE SAVED !!!')
# else:
#     print('~~~ model with NOT be saved ~~~')

# if check_model:
#     print('---- Script will terminate after one epoch ----')
#     nepoch = 1

# for debugging
save_model, check_model, is_eosin, is_inflam, nfill = True, True, False, True, 1
lr, p, nepoch, batch = 1e-3, 64, 1, 2

from cells import valid_cells, inflam_cells

# Needs to be mutually exlusive
assert is_eosin != is_inflam
if is_eosin:
    cells = ['eosinophil']
else:
    cells = inflam_cells

cell_fold = 'inflam' if is_inflam else 'eosin'

print('Cells: %s\nnepoch: %i\nbatch: %i\nlr: %0.3f, p: %i' % (cells, nepoch, batch, lr, p))

# number of padded points (i.e. count inflator)
fillfac = (2 * nfill + 1) ** 2
print('nfill: %i, fillfac: x%i' % (nfill, fillfac))
# Number of channels from baseline
max_channels = p*2**4
print('Baseline: %i, maximum number of channels: %i' % (p, max_channels))

# import os
# import gc
# import hickle

# import pandas as pd
# import plotnine as pn
# from funs_support import sigmoid, find_dir_cell, makeifnot, hash_hp, write_pickle
# from funs_plotting import gg_save
# from funs_stats import get_YP, cross_entropy, global_auprc, global_auroc

import sys
from time import time
import random
import numpy as np
from mdls.unet import UNet
import torch
# from funs_torch import CellCounterDataset, img2tensor, randomRotate, randomFlip
# from torchvision import transforms
# from torch.utils import data
# from torch.nn import DataParallel

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
    n_cuda = torch.cuda.device_count()
    cuda_index = list(range(n_cuda))
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
    n_cuda, cuda_index = None, None
device = torch.device('cuda:0' if use_cuda else 'cpu')

# # Set up folders
# dir_base = find_dir_cell()
# dir_output = os.path.join(dir_base, 'output')
# dir_figures = os.path.join(dir_output, 'figures')
# lst_dir = [dir_output, dir_figures]
# assert all([os.path.exists(z) for z in lst_dir])
# dir_checkpoint = os.path.join(dir_output, 'checkpoint')
# dir_cell = os.path.join(dir_checkpoint, cell_fold)

# Image parameters
n_channels = 3
pixel_max = 255
b0 = -6.0

###################################
## --- (1) INITIALIZE MODELS --- ##

seednum = 1234
if use_cuda:
    torch.cuda.manual_seed_all(seednum)
torch.manual_seed(seednum)
random.seed(seednum)
np.random.seed(seednum)

# Load the model
mdl = UNet(n_channels=n_channels, n_classes=1, bl=p, batchnorm=True)
# Set as float64
mdl.float()
with torch.no_grad():
    mdl.outc.conv.bias.fill_(b0)
# # Enable data parallelism if possible
# if n_cuda is not None:
#     if n_cuda > 1:
#         mdl = DataParallel(mdl)
mdl.to(device)
# Check CUDA status for model
print('Are network parameters cuda?: %s' %
      all([z.is_cuda for z in mdl.parameters()]))

# # Binary loss
# criterion = torch.nn.BCEWithLogitsLoss()
# # Optimizer
# optimizer = torch.optim.Adagrad(params=mdl.parameters(), lr=lr)

###################################
## --- (2) CHECK RANDOM DATA --- ##

mdl.eval()
niter = 10
stime = time()
for ii in range(niter):
    timg_ii = torch.rand(1, 3, 501, 501).float().to(device)
    logits_ii = mdl(timg_ii).mean()
    dtime, nleft = time() - stime, niter - (ii+1)
    rate = (ii + 1) / dtime
    seta = nleft / rate
    print('ETA: %.1f seconds' % seta)

sys.exit('Stop here')


###########################
## --- (1) LOAD DATA --- ##

# --- (i) Train/Val/Test IDs --- #
df_tt = pd.read_csv(os.path.join(dir_output,'train_val_test.csv'))
di_tt = df_tt.groupby('tt').apply(lambda x: x.idt_tissue.to_list()).loc[['train','val','test']].to_dict()
idt_tissue = df_tt.idt_tissue.to_list()
# Corresponding index
di_tt_idx = {k: np.where(pd.Series(idt_tissue).isin(v))[0] for k,v in di_tt.items()}

# --- (ii) Load data --- #
# Images ['img'] and labels (gaussian blur) ['lbls']
path_pickle = os.path.join(dir_output, 'annot_hsk.pickle')
di_data = hickle.load(path_pickle)
gc.collect()
# Aggregate counts
df_cells = pd.read_csv(os.path.join(dir_output,'df_cells.csv'))
# Sum based on cell type
df_cells = df_cells[['ds','idt_tissue']].assign(cell=df_cells[cells].sum(1).values)

# --- (iii) Set labels to match cell type --- #
idx_cell = np.where(pd.Series(valid_cells).isin(cells))[0]
Ymat_lbls = np.stack([di_data[idt]['lbls'] for idt in idt_tissue],0)
Ymat_lbls = Ymat_lbls[:, :, :, idx_cell]
Ymat_lbls = np.apply_over_axes(np.sum, Ymat_lbls, 3)
Ymat_img = np.stack([di_data[idt]['img'] for idt in idt_tissue],0)
df_est = pd.DataFrame({'idt_tissue':idt_tissue,'est':np.squeeze(np.apply_over_axes(np.sum, Ymat_lbls, [1,2,3])) / fillfac})
df_est = df_cells.merge(df_est).assign(aepct=lambda x: np.abs(x.cell/x.est-1))
assert df_est.aepct.max() < 0.02

# --- (iv) Check image/label size concordance --- #
assert Ymat_lbls.shape[1:3] == Ymat_img.shape[1:3]
n_pixels = Ymat_lbls.shape[1]

# --- (v) Get mean number of cells/pixels for intercept initialization --- #
# Use only training data to initialize intercept
mu_pixels = np.mean(np.apply_over_axes(np.mean, Ymat_lbls[di_tt_idx['train']],[1,2,3]).flatten())
mu_cells = df_cells[df_cells.idt_tissue.isin(di_tt['train'])].cell.mean()
err = 100 * ((mu_pixels * n_pixels**2 / fillfac) / mu_cells - 1)
print('Error: %.2f%%' % err)
b0 = np.log(mu_pixels / (1 - mu_pixels))


#################################
## --- (3) CHECK INTERCEPT --- ##

mdl.eval()
n_image = len(Ymat_img)
stime = time()
mat = np.zeros([n_image, 2])
for ii in range(n_image):
    timg_ii = torch.tensor(np.expand_dims(Ymat_img[ii].transpose(2,0,1), 0) / 255).to(device)
    with torch.no_grad():
        logits_ii = mdl(timg_ii)
    mu_sig = torch.sigmoid(logits_ii).mean().cpu().detach().numpy()
    mu_logit = logits_ii.mean().cpu().detach().numpy()
    mat[ii] = [mu_logit, mu_sig]
    dtime, nleft = time() - stime, n_image - (ii+1)
    rate = (ii + 1) / dtime
    seta = nleft / rate
    print('ETA: %.1f seconds' % seta)
torch.cuda.empty_cache()
mat[:,0].mean()
mat[:,1].sum()
