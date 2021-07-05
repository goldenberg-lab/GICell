import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_model', dest='save_model', action='store_true', help='Save model as .pt file')
parser.add_argument('--is_eosin', dest='is_eosin', action='store_true', help='Eosinophil cell only')
parser.add_argument('--is_inflam', dest='is_inflam', action='store_true', help='Eosinophil + neutrophil + plasma + lymphocyte')
parser.set_defaults(is_eosin=False, is_inflam=False)
parser.add_argument('--nepoch', type=int, default=110, help='Number of epochs')
parser.add_argument('--batch', type=int, default=1, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--p', type=int, default=8, help='Number of initial params for NNet')
parser.add_argument('--nfill', type=int, default=1, help='How many points to pad around pixel annotation point')
args = parser.parse_args()
save_model, is_eosin, is_inflam = args.save_model, args.is_eosin, args.is_inflam
nepoch, batch, lr, p, nfill = args.nepoch, args.batch, args.lr, args.p, args.nfill

if save_model:
    print('!!! WARNING --- MODEL WILL BE SAVED !!!')
else:
    print('~~~ model with NOT be saved ~~~')

# # for debugging
# save_model, is_eosin, is_inflam, nfill = True, False, True, 1
# lr, p, nepoch, epoch_check, batch = 1e-3, 16, 2, 1, 2

# Needs to be mutually exlusive
assert is_eosin != is_inflam
if is_eosin:
    cells = ['eosinophil']
else:
    cells = ['eosinophil','neutrophil','plasma','lymphocyte']

cell_fold = 'inflam' if is_inflam else 'eosin'

print('Cells: %s\nnepoch: %i\nbatch: %i\nlr: %0.3f, p: %i' % (cells, nepoch, batch, lr, p))

# number of padded points (i.e. count inflator)
fillfac = (2 * nfill + 1) ** 2
print('nfill: %i, fillfac: x%i' % (nfill, fillfac))
# Number of channels from baseline
max_channels = p*2**4
print('Baseline: %i, maximum number of channels: %i' % (p, max_channels))

import os
import pickle
import hickle
import gc
import numpy as np
import random
import pandas as pd
from funs_support import sigmoid, find_dir_cell, makeifnot
from funs_plotting import gg_save
from funs_stats import get_YP, cross_entropy, global_auprc, global_auroc
from time import time
import plotnine as pn

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

# Order of valid_cells matters (see idx_cells & label_blur)
valid_cells = ['eosinophil', 'neutrophil', 'plasma', 'enterocyte', 'other', 'lymphocyte']

# Image parameters
n_channels = 3
pixel_max = 255

###########################
## --- (1) LOAD DATA --- ##

# --- (i) Train/Val/Test IDs --- #
df_tt = pd.read_csv(os.path.join(dir_output,'train_val_test.csv'))
di_tt = df_tt.groupby('tt').apply(lambda x: x.idt_tissue.to_list()).loc[['train','val','test']].to_dict()
idt_tissue = df_tt.idt_tissue.to_list()

# --- (ii) Load data --- #
# Images ['img'] and labels (gaussian blur) ['lbls']
path_pickle = os.path.join(dir_output, 'annot_hsk.pickle')
di_data = hickle.load(path_pickle)
gc.collect()
# Aggregate counts
df_cells = pd.read_csv(os.path.join(dir_output,'df_cells.csv'))

# --- (iii) Set labels to match cell type --- #
idx_cell = np.where(pd.Series(valid_cells).isin(cells))[0]
holder = []
for kk, idt in enumerate(idt_tissue):
    tmp = di_data[idt]['lbls'].copy()
    tmp2 = np.atleast_3d(tmp[:, :, idx_cell].sum(2))
    gt = int(df_cells.query('idt_tissue==@idt',engine='python')[cells].sum(1).values)
    est = tmp2.sum() / fillfac
    if gt > 0:
        err_pct = 100*(gt / est - 1)
        assert np.abs(err_pct) < 2
    di_data[idt]['lbls'] = tmp2
    holder.append(err_pct)
    del tmp, tmp2, gt
dat_err_pct = pd.DataFrame({'idt':idt_tissue, 'err':holder}).assign(err=lambda x: x.err.abs())
dat_err_pct = dat_err_pct.sort_values('err',ascending=False).round(2).reset_index(None,True)
print(dat_err_pct.head())

# --- (iv) Check image/label size concordance --- #
dat_imsize = pd.DataFrame([di_data[idt]['lbls'].shape[:2] + di_data[idt]['img'].shape[:2] for idt in idt_tissue])
n_pixels = dat_imsize.iloc[0,0]
assert np.all(dat_imsize == n_pixels)

# --- (v) Get mean number of cells/pixels for intercept initialization --- #
# Use only training data to initialize intercept
mu_pixels = np.mean([di_data[z]['lbls'].mean() for z in di_data if z in di_tt['train']])
mu_cells = df_cells.query('idt_tissue.isin(@idt_tissue)',engine='python')[cells].sum(1).mean()
err = 100 * ((mu_pixels * n_pixels**2 / fillfac) / mu_cells - 1)
print('Error: %.2f%%' % err)
b0 = np.log(mu_pixels / (1 - mu_pixels))

# Compare percent of non-empty pixels to actual count
tmp1 = np.array([np.mean(di_data[z]['lbls']!=0) for z in idt_tissue])
tmp2 = np.array([np.round(np.sum(di_data[z]['lbls'])/fillfac).astype(int) for z in idt_tissue])
tmp3 = pd.DataFrame({'idt':idt_tissue, 'pct':tmp1, 'est':tmp2})
tmp4 = df_cells.rename(columns={'idt_tissue':'idt'})[['idt'] + cells]
tmp4 = tmp4.drop(columns=cells).assign(n=tmp4[cells].sum(1))
dat_count_pct = tmp4.merge(tmp3, 'right', 'idt').sort_values('pct',ascending=False)
dat_count_pct = dat_count_pct.drop(columns='idt').set_index(dat_count_pct.idt)
assert np.abs(dat_count_pct.n - dat_count_pct.est).max() <= 1
gg_count_pct = (pn.ggplot(dat_count_pct,pn.aes(x='n',y='pct')) + pn.theme_bw() + pn.geom_point(size=0.5))
tmp_fn = 'gg_count_pct_' + cell_fold + '.png'
gg_save(tmp_fn, dir_figures, gg_count_pct, 5, 4)
# Ensure relationship is at least 99% correlated
assert dat_count_pct[['n','pct']].corr().iloc[0,1] > 0.99


###################################
## --- (2) INITIALIZE MODELS --- ##

seednum = 1234
if use_cuda:
    torch.cuda.manual_seed_all(seednum)
torch.manual_seed(seednum)
random.seed(seednum)
np.random.seed(seednum)

# Load the model
mdl = UNet(n_channels=3, n_classes=1, bl=p, batchnorm=True)
with torch.no_grad():
    mdl.outc.conv.bias.fill_(b0)
# Enable data parallelism if possible
if n_cuda is not None:
    if n_cuda > 1:
        mdl = DataParallel(mdl)
mdl.to(device)
# Check CUDA status for model
print('Are network parameters cuda?: %s' %
      all([z.is_cuda for z in mdl.parameters()]))

# Binary loss
criterion = torch.nn.BCEWithLogitsLoss()
# Optimizer
optimizer = torch.optim.Adagrad(params=mdl.parameters(), lr=lr)

# Check that intercept approximates cell count
tnow = time()
mat = np.zeros([len(idt_tissue), 2])
mdl.eval()
for ii, idt in enumerate(idt_tissue):
    if (ii + 1) % 25 == 0:
        print('ID-tissue %s (%i of %i)' % (idt, ii + 1, len(idt_tissue)))
    tens = img2tensor(device)([di_data[idt]['img'],di_data[idt]['lbls']])[0]
    tens = tens.reshape([1, n_channels, n_pixels, n_pixels]) / pixel_max
    with torch.no_grad():
        logits = mdl(tens)
        ncl = logits.cpu().mean().numpy()+0
        nc = torch.sigmoid(logits).cpu().sum().numpy()+0
        mat[ii] = [nc, ncl]
torch.cuda.empty_cache()
print('Took %i seconds to pass through all images' % (time() - tnow))
emp_cells, emp_ncl = mat.mean(0)
print('Intercept: %.2f, empirical logits: %.2f' % (b0, emp_ncl))
print('fillfac*cells: %.2f, predicted cells: %.2f' % (mu_cells*fillfac, emp_cells))

##############################
## --- (3) DATA LOADERS --- ##

# Check no overlap
print('--- Sample sizes ---')
print({print('%s = %i' % (k,len(v))) for k, v in di_tt.items()})

# Create datasetloader class
train_params = {'batch_size': batch, 'shuffle': True}
val_params = {'batch_size': 1, 'shuffle': False}
eval_params = {'batch_size': 1,'shuffle': False}

multiclass = False

# Training (random rotations and flips)
train_transform = transforms.Compose([randomRotate(tol=1e-4), randomFlip(), img2tensor(device)])
train_data = CellCounterDataset(di=di_data, ids=di_tt['train'], transform=train_transform, multiclass=multiclass)
train_gen = data.DataLoader(dataset=train_data,**train_params)

# Validation
val_transform = transforms.Compose([img2tensor(device)])
val_data = CellCounterDataset(di=di_data, ids=di_tt['val'], transform=val_transform, multiclass=multiclass)
val_gen = data.DataLoader(dataset=val_data,**val_params)

# Eval (all sample)
eval_data = CellCounterDataset(di=di_data, ids=di_tt['train'] + di_tt['val'],
                               transform=val_transform, multiclass=multiclass)
eval_gen = data.DataLoader(dataset=eval_data, **eval_params)


# Iniatize Y
Y_val = get_YP(val_gen, mdl, n_pixels, n_pixels, ret_Y=True, ret_P=False)
Ybin_val = np.where(Y_val>0,1,0)
gt1 = np.round(Y_val.sum(0).sum(0) / fillfac).astype(int)
gt2 = dat_count_pct.loc[di_tt['val']]['n'].values
assert np.all(gt1 == gt2)

Phat_init = sigmoid(get_YP(val_gen, mdl, n_pixels, n_pixels, ret_Y=False, ret_P=True))
print('Initial AUC: %.1f%%' % (100*global_auroc(Ybin_val, Phat_init)))

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
    for ids_batch, lbls_batch, imgs_batch in train_gen:
        ii += 1
        ids_batch = list(ids_batch)
        nbatch = len(ids_batch)
        # --- Forward pass --- #
        optimizer.zero_grad()
        logits = mdl(imgs_batch)
        assert logits.shape == lbls_batch.shape
        loss = criterion(input=logits,target=lbls_batch)
        # --- Backward pass --- #
        loss.backward()
        # --- Gradient step --- #
        optimizer.step()
        # --- Performance --- #
        ii_loss = loss.item()
        # assert np.abs(ii_loss - cross_entropy(np.squeeze(t2n(lbls_batch)), sigmoid(np.squeeze(t2n(logits))))) < 1e-8
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
    P_val = sigmoid(get_YP(val_gen, mdl, n_pixels, n_pixels, ret_Y=False, ret_P=True))
    # Calculate cross entropy, AUROC, precision, and recall on validation
    ce_val = cross_entropy(Y_val, P_val)
    auc_val = global_auroc(Ybin_val, P_val)
    pr_val = global_auprc(Ybin_val, P_val, n_points=27)
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
dat_ce_auc = pd.concat(holder_ce_auc).reset_index(None, True)
dat_pr = pd.concat(holder_pr).reset_index(None, True)

# Hash all hyperparameters
df_slice = pd.DataFrame({'lr':lr, 'p':p, 'batch':batch},index=[0])
df_slice = df_slice.loc[0].reset_index().rename(columns={'index':'hp',0:'val'})
hp_string = pd.Series([df_slice.apply(lambda x: x[0] + '=' + str(x[1]), 1).str.cat(sep='_')])
code_hash = pd.util.hash_array(hp_string)[0]

# Pickle the dictionary
di = {'hp':df_slice, 'ce_auc':dat_ce_auc, 'pr':dat_pr}
if save_model:
    di['mdl'] = mdl

path_di = os.path.join(dir_cell, str(code_hash) + '.pkl')
with open(path_di, 'wb') as handle:
    pickle.dump(di, handle, protocol=pickle.HIGHEST_PROTOCOL)
