import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_model', dest='save_model', action='store_true', help='Save model as .pt file')
parser.add_argument('--check_int', dest='check_int', action='store_true', help='Should forward pass be done to print intercept?')
parser.add_argument('--check_model', dest='check_model', action='store_true', help='Stop model after one epoch')
parser.add_argument('--is_eosin', dest='is_eosin', action='store_true', help='Eosinophil cell only')
parser.add_argument('--is_inflam', dest='is_inflam', action='store_true', help='Eosinophil + neutrophil + plasma + lymphocyte')
parser.set_defaults(is_eosin=False, is_inflam=False)
parser.add_argument('--ds_test', nargs='+', help='Folders that should be reserved for testing')
parser.add_argument('--nepoch', type=int, default=110, help='Number of epochs')
parser.add_argument('--batch', type=int, default=1, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--p', type=int, default=8, help='Number of initial params for NNet')
parser.add_argument('--nfill', type=int, default=1, help='How many points to pad around pixel annotation point')
args = parser.parse_args()
save_model, check_int, check_model = args.save_model, args.check_int, args.check_model
is_eosin, is_inflam, ds_test = args.is_eosin, args.is_inflam, args.ds_test
nepoch, batch, lr, p, nfill = args.nepoch, args.batch, args.lr, args.p, args.nfill
print('args = %s' % args)


if save_model:
    print('!!! WARNING --- MODEL WILL BE SAVED !!!')
else:
    print('~~~ model with NOT be saved ~~~')

if check_model:
    print('---- Script will terminate after one epoch ----')
    nepoch = 1

# # for debugging
# save_model, check_int, check_model = False, True, True
# is_eosin, is_inflam, nfill = True, False, 1
# ds_test='oscar dua 70608'.split(' ')
# lr, p, nepoch, batch = 1e-3, 16, 1, 2

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

import os
import sys
import gc
import hickle
import random
import numpy as np
import pandas as pd
from time import time
from funs_support import sigmoid, find_dir_cell, makeifnot, hash_hp, write_pickle, t2n
from funs_stats import get_YP, cross_entropy, global_auprc, global_auroc

from mdls.unet import UNet
import torch
from funs_torch import CellCounterDataset, img2tensor, randomRotate, randomFlip
from torchvision import transforms
from torch.utils import data
from torch.nn import DataParallel

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
    n_cuda = torch.cuda.device_count()
    cuda_index = list(range(n_cuda))
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
    n_cuda, cuda_index = None, None
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Set up folders
dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')
lst_dir = [dir_output, dir_figures]
assert all([os.path.exists(z) for z in lst_dir])
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
dir_cell = os.path.join(dir_checkpoint, cell_fold)

lst_newdir = [dir_checkpoint, dir_cell]
for dir in lst_newdir:
    makeifnot(dir)

# Image parameters
n_channels = 3
pixel_max = 255

# Tolerance for actual/expected
tol_pct, tol_dcell = 0.02, 2


###########################
## --- (1) LOAD DATA --- ##

# --- (i) Train/Val/Test IDs --- #
df_tt = pd.read_csv(os.path.join(dir_output,'train_val_test.csv'))
u_tt = list(df_tt[df_tt['tt'] != 'test']['tt'].unique())
ds_trainval = list(df_tt[df_tt['tt'].isin(u_tt)]['ds'].unique())
di_tt = pd.DataFrame({'tt':df_tt['tt'],'ds_idt':df_tt.apply(lambda x: x[['ds','idt']].to_list(),axis=1)})
di_tt = di_tt.groupby('tt').apply(lambda x: x.ds_idt.to_list()).to_dict()

# --- (ii) Load training/validation data --- #
# Images ['img'] and labels (gaussian blur) ['lbls']
tmp_di = {}
for ds in ds_trainval:
    path_pickle = os.path.join(dir_output, 'annot_%s.pickle' % ds)
    tmp_di[ds] = hickle.load(path_pickle)
# Invert the key order from [ds][idt] to [set][idt]
di_data = {}
for tt in u_tt:
    di_data[tt] = {}
    for ds, idt in di_tt[tt]:
        di_data[tt][idt] = {}
        di_data[tt][idt][ds] = tmp_di[ds][idt]
del tmp_di

# --- (iii) Aggregate cell counts --- #
cn_idx = ['ds','idt']
df_cells = pd.read_csv(os.path.join(dir_output,'df_cells.csv'))
df_cells.drop(columns=['h','w'], inplace=True)
# Sum based on cell type
df_cells = df_cells[cn_idx+cells].assign(cell=df_cells[cells].sum(1).values)
df_cells.drop(columns=cells, inplace=True)

# --- (iv) Set labels to match cell type --- #
idx_cell = np.where(pd.Series(valid_cells).isin(cells))[0]
holder = []
for tt in di_data:
    for idt in di_data[tt].keys():
        for ds in di_data[tt][idt].keys():
            tmp = di_data[tt][idt][ds]['lbls'].copy()
            tmp = np.atleast_3d(tmp[:, :, idx_cell].sum(2))
            gt = df_cells.query('idt == @idt')['cell'].values[0]
            est = tmp.sum() / fillfac
            if gt > 0:
                err_pct = 100*np.abs(gt / est - 1)
                err_dcell = np.abs(gt - est)
                assert (err_pct < 100*tol_pct) or (err_dcell < tol_dcell)
                tmp_err = pd.DataFrame({'ds':ds, 'idt':idt, 'pct':err_pct, 'dcell':err_dcell},index=[0])
                holder.append(tmp_err)
            di_data[tt][idt][ds]['lbls'] = tmp        
            del tmp, gt, est
dat_err_pct = pd.concat(holder).sort_values('pct',ascending=False).round(2)
dat_err_pct.reset_index(None,drop=True,inplace=True)
print(dat_err_pct.head())

# --- (v) Get mean number of cells/pixels for intercept initialization --- #
holder = []
for idt in di_data['train'].keys():
    for ds in di_data['train'][idt].keys():
        tmp_lbls = di_data['train'][idt][ds]['lbls'].copy()
        mu_cells = tmp_lbls.sum() / np.prod(tmp_lbls.shape[:2])
        tmp_mu = pd.DataFrame({'ds':ds, 'idt':idt, 'mu':mu_cells},index=[0])
        holder.append(tmp_mu)
dat_mu_cells = pd.concat(holder).reset_index(None, drop=True)
b0 = dat_mu_cells['mu'].mean()
# Do logit transformation
b0 = np.log(b0 / (1 - b0))


###################################
## --- (2) INITIALIZE MODELS --- ##

seednum = 1234
if use_cuda:
    torch.cuda.manual_seed_all(seednum)
torch.manual_seed(seednum)
random.seed(seednum)
np.random.seed(seednum)

# Load the model
mdl = UNet(n_channels=n_channels, n_classes=1, bl=p, batchnorm=True)
with torch.no_grad():
    mdl.outc.conv.bias.fill_(b0)
# Enable data parallelism if possible
if n_cuda is not None:
    if n_cuda > 1:
        mdl = DataParallel(mdl)
mdl.to(device)
# Check CUDA status for model
print('Are network parameters cuda?: %s' % all([z.is_cuda for z in mdl.parameters()]))

# Binary loss
criterion = torch.nn.BCEWithLogitsLoss()
# Optimizer
optimizer = torch.optim.Adagrad(params=mdl.parameters(), lr=lr)

if check_int:  # Check that intercept approximates cell count
    tnow = time()
    mdl.eval()
    enc_tens = img2tensor(device)
    holder, ii = [], 0
    for idt in di_data['train'].keys():
        for ds in di_data['train'][idt].keys():
            ii += 1
            if ii % 25 == 0:
                print('ID-tissue %s (%i)' % (idt, ii))
            tmp_di = di_data['train'][idt][ds].copy()
            tens = enc_tens([tmp_di['img'], tmp_di['lbls']])[0]
            # First channel should be batch size
            tens = torch.unsqueeze(tens, dim=0).float() / pixel_max
            with torch.no_grad():
                logits = t2n(mdl(tens))
            ncl = logits.mean()
            nc = sigmoid(logits).sum()
            mu = nc / np.prod(tens.shape[2:])
            tmp_pred = pd.DataFrame({'ds':ds, 'idt':idt, 'mu':mu, 'ncl':ncl}, index=[ii])
            holder.append(tmp_pred)
    torch.cuda.empty_cache()
    print('Took %i seconds to pass through all images' % (time() - tnow))
    dat_b0 = pd.concat(holder)
    emp_cells, emp_ncl = dat_b0[['mu','ncl']].mean(0)
    print('Intercept: %.3f, empirical logits: %.3f' % (b0, emp_ncl))
    print('Pixels per cell: %.1f, sigmoid: %.1f' % (1/mu_cells, 1/emp_cells))
# Clean out any temporary files
gc.collect()

##############################
## --- (3) DATA LOADERS --- ##

# Check no overlap
print('--- Sample sizes ---')
print({print('%s = %i' % (k,len(v))) for k, v in di_tt.items()})

# Create datasetloader class
train_params = {'batch_size':batch, 'shuffle':True}
val_params = {'batch_size':1, 'shuffle':False}
eval_params = {'batch_size':1,'shuffle':False}

multiclass = False

# Training (random rotations and flips)
train_transform = transforms.Compose([randomRotate(), randomFlip(), img2tensor(device)])
train_data = CellCounterDataset(di=di_data['train'], transform=train_transform, multiclass=multiclass)
train_gen = data.DataLoader(dataset=train_data, **train_params)

# Validation
val_transform = transforms.Compose([img2tensor(device)])
val_data = CellCounterDataset(di=di_data['val'], transform=val_transform, multiclass=multiclass)
val_gen = data.DataLoader(dataset=val_data,**val_params)

# Iniatize Y
idx_Yval, Yval = get_YP(val_gen, mdl, ret_Y=True, ret_P=False, ret_idx=True)
bin_Yval = np.where(np.isnan(Yval), np.nan, np.where(Yval > 0, 1, 0))
gt1 = np.ceil(np.nansum(np.nansum(Yval,1),1) / fillfac).astype(int)
gt2 = idx_Yval.merge(df_cells,'left')['cell']
assert np.all(gt1 == gt2), 'Yval does not align with df_cells'
Phat_init = sigmoid(get_YP(val_gen, mdl, ret_Y=False, ret_P=True))
pwauc_init = 100*global_auroc(bin_Yval, Phat_init)
print('Initial AUC (pixel-wise): %.1f%%' % pwauc_init)


##########################
## --- (4) TRAINING --- ##

b_check = (len(train_gen) + 1) // 5

stime, ee, ii = time(), 0, 1
holder_ce_auc, holder_pr = [], []
for ee in range(nepoch):
    print('--------- EPOCH %i of %i ----------' % (ee+1, nepoch))

    ### --- MODEL TRAINING --- ###
    mdl.train()
    lst_ce, lst_pred_act, lst_ids = [], [], []
    ii = 0
    for ds_batch, ids_batch, lbls_batch, imgs_batch in train_gen:
        ii += 1
        ids_batch = list(ids_batch)
        nbatch = len(ids_batch)
        # --- Forward pass --- #
        optimizer.zero_grad()
        logits = mdl(imgs_batch)
        assert logits.shape == lbls_batch.shape
        # Flatten the values to allow for removal of nan-values
        logits = logits.flatten()
        lbls_batch = lbls_batch.flatten()
        idx_keep = ~torch.isnan(lbls_batch)
        logits = logits[idx_keep]
        lbls_batch = lbls_batch[idx_keep]
        loss = criterion(input=logits, target=lbls_batch)
        # --- Backward pass --- #
        loss.backward()
        # --- Gradient step --- #
        optimizer.step()
        # --- Performance --- #
        ii_loss = loss.item()
        # Empty cache
        torch.cuda.empty_cache()
        del lbls_batch, imgs_batch
        lst_ce.append(ii_loss)
        lst_ids.append(ids_batch)
        if (ii + 1) % b_check == 0:
            print('Batch %i of %i: %s' % (ii+1, len(train_gen), ', '.join(ids_batch)))
            print('Cross-entropy loss: %0.6f' % ii_loss)
    ce_train = np.mean(lst_ce)
    print('--- End of epoch %i, CE=%.3f' % (ee+1, ce_train*1e3))

    ### --- MODEL EVALUATION --- ###
    # Cross-entropy, precision, recall
    mdl.eval()
    # Get the probability dist
    P_val = sigmoid(get_YP(val_gen, mdl, ret_Y=False, ret_P=True))
    # Calculate cross entropy, AUROC, precision, and recall on validation
    ce_val = cross_entropy(Yval, P_val)
    auc_val = global_auroc(bin_Yval, P_val)
    pr_val = global_auprc(bin_Yval, P_val, n_points=27)
    pr_val.insert(0,'epoch', ee+1)
    tmp_ce_auc = pd.DataFrame({'epoch':ee+1, 'auc':auc_val, 'ce':ce_val},index=[ee])
    # Save
    holder_ce_auc.append(tmp_ce_auc)
    holder_pr.append(pr_val)
    # Calculate the ETA
    dsec, nleft = time() - stime, nepoch - (ee + 1)
    rate = (ee + 1) / dsec
    seta = nleft / rate
    meta = seta / 60
    print('--- ETA: %.1f minutes (%i of %i) ---' % (meta, ee+1, nepoch))


#########################
## --- (5) EX POST --- ##

# Merge dataframes
dat_ce_auc = pd.concat(holder_ce_auc).reset_index(None, drop=True)
dat_pr = pd.concat(holder_pr).reset_index(None, drop=True)

# Hash all hyperparameters
cn_hp = ['lr', 'p', 'batch', 'nepoch']
df_slice = pd.DataFrame({'lr':lr, 'p':p, 'batch':batch, 'nepoch':nepoch},index=[0])
code_hash = hash_hp(df_slice, cn_hp)

# Pickle the dictionary
di = {'code_hash':code_hash, 'hp':df_slice, 'ce_auc':dat_ce_auc, 'pr':dat_pr}

if check_model:
    sys.exit('check_model exit')

if save_model:
    di['mdl'] = mdl

path_di = os.path.join(dir_cell, str(code_hash) + '.pkl')
write_pickle(di, path_di)
