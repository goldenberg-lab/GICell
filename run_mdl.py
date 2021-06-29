import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--is_eosin', dest='is_eosin', action='store_true', help='Eosinophil cell only')
parser.add_argument('--is_inflam', dest='is_inflam', action='store_true', help='Eosinophil + neutrophil + plasma + lymphocyte')
parser.set_defaults(is_eosin=False, is_inflam=False)
parser.add_argument('--nepoch', type=int, default=1000, help='Number of epochs')
parser.add_argument('--batch', type=int, default=1, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--p', type=int, default=8, help='Number of initial params for NNet')
parser.add_argument('--nfill', type=int, default=1, help='How many points to pad around pixel annotation point')
args = parser.parse_args()
is_eosin, is_inflam = args.is_eosin, args.is_inflam
nepoch, batch, lr, p, nfill = args.nepoch, args.batch, args.lr, args.p, args.nfill

# # for debugging
# is_eosin, is_inflam, nfill = True, False, 1
# lr, p, nepoch, epoch_check, batch = 1e-3, 16, 2, 1, 2

# Needs to be mutually exlusive
assert is_eosin != is_inflam
if is_eosin:
    cells = ['eosinophil']
else:
    cells = ['eosinophil','neutrophil','plasma','lymphocyte']

print('Cells: %s\nnepoch: %i\nbatch: %i\nlr: %0.3f, np: %i' % (cells, nepoch, batch, lr, p))

# number of padded points (i.e. count inflator)
fillfac = (2 * nfill + 1) ** 2
print('nfill: %i, fillfac: x%i' % (nfill, fillfac))
# Number of channels from baseline
max_channels = p*2**4
print('Baseline: %i, maximum number of channels: %i' % (p, max_channels))

import os
import hickle
import gc
import numpy as np
import random
import pandas as pd
from funs_support import sigmoid, makeifnot, t2n, find_dir_cell
from funs_plotting import gg_save
from time import time
import torch
from mdls.unet import UNet
from sklearn.metrics import r2_score
from datetime import datetime
from funs_torch import CellCounterDataset, img2tensor, randomRotate, randomFlip
from torchvision import transforms
from torch.utils import data
# from plotnine import *

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device('cuda' if use_cuda else 'cpu')

# Hyperparameter configs
df_slice = pd.DataFrame({'lr':lr, 'np':np,'nepoch':nepoch,'batch':batch},index=[0])

# Get current day
dnow = datetime.now().strftime('%Y_%m_%d')

# Set up folders
dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')
lst_dir = [dir_output, dir_figures]
assert all([os.path.exists(z) for z in lst_dir])
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
dir_cell = os.path.join(dir_checkpoint, '_'.join(np.sort(cells)))
dir_datecell = os.path.join(dir_cell, dnow)
# Folder for hyperparameter configuration
hp = df_slice.T[0].astype(str).str.cat(sep='').replace('.','')
dir_hp = os.path.join(dir_datecell,hp)

# lst_newdir = [dir_checkpoint, dir_cell, dir_datecell, dir_hp]
# for dir in lst_newdir:
#     makeifnot(dir)

# Order of valid_cells matters (see idx_cells & label_blur)
valid_cells = ['eosinophil', 'neutrophil', 'plasma', 'enterocyte', 'other', 'lymphocyte']

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
    gt = int(df_cells.query('idt_tissue==@idt',engine='python')[cells].values)
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
err = 100 * (mu_pixels * n_pixels**2 / fillfac / mu_cells - 1)
print('Error: %.2f%%' % err)
b0 = np.log(mu_pixels / (1 - mu_pixels))

# Compare percent of non-empty pixels to actual count
tmp1 = np.array([np.mean(di_data[z]['lbls']!=0) for z in di_data if z in idt_tissue])
tmp2 = pd.DataFrame({'idt_tissue':idt_tissue, 'pct':tmp1})
dat_count_pct = df_cells[['ds','idt_tissue'] + cells].merge(tmp2).rename(columns={'idt_tissue':'idt'})
dat_count_pct = dat_count_pct.drop(columns=cells).assign(n=dat_count_pct[cells].sum(1))

from plotnine import *
gg_count_pct = (ggplot(dat_count_pct,aes(x='n',y='pct')) + theme_bw() + 
    geom_point(size=0.5))
gg_save('gg_count_pct.png', dir_figures, gg_count_pct, 5, 4)



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
mdl.to(device)
with torch.no_grad():
    mdl.outc.conv.bias.fill_(b0)
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


##########################
## --- (3) TRAINING --- ##

# Check no overlap
print('--- Sample sizes ---')
print({print('%s = %i' % (k,len(v))) for k, v in di_tt.items()})

# Create datasetloader class
train_params = {'batch_size': batch, 'shuffle': True}
val_params = {'batch_size': batch,'shuffle': False}
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


tnow = time()
epoch_loss = []
ee, ii = 0, 1
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
        ii_loss = float(loss)
        # Empty cache
        del lbls_batch, imgs_batch
        torch.cuda.empty_cache()
        lst_ce.append(ii_loss)
        lst_ids.append(ids_batch)
        if (ii + 1) % 25 == 0:
            print('Batch %i of %i: %s' % (ii+1, len(train_gen), ', '.join(ids_batch)))
            print('Cross-entropy loss: %0.6f' % ii_loss)
    ce_train = np.mean(lst_ce)
    print('--- End of epoch %i, CE=%.3f' % (ee, ce_train*1e3))

    ### --- MODEL EVALUATION --- ###
    # Cross-entropy, precision, recall
    mdl.eval()
    holder_eval = []
    for ii, batch_ii in enumerate(eval_gen):
        ids_batch, lbls_batch, imgs_batch = batch_ii
        if (ii + 1) % 25 == 0:
            print('Prediction for: %s' % ', '.join(ids_batch))
        lbls = np.stack([di_data[ids]['lbls'] for ids in ids_batch], 3)
        #print('%.1f%%' % (np.mean(lbls != 0)*100))
        if np.mean(lbls != 0):
            break

        with torch.no_grad():
            logits = t2n(mdl(imgs_batch))
            ce = float(t2n(criterion(input=mdl(imgs_batch), target=lbls_batch)))
        logits = logits.transpose(2, 3, 1, 0)
        # Drop 1-d axes
        logits, lbls = np.squeeze(logits), np.squeeze(lbls)
        assert lbls.shape == logits.shape == (n_pixels, n_pixels)
        # Calculate average precision recall

        ids_seq = list(ids_batch)
        pred_seq = phat.sum(0).sum(0).sum(0) / fillfac
        act_seq = gaussian.sum(0).sum(0).sum(0) / fillfac
        tmp = pd.DataFrame({'ids': ids_seq, 'pred': pred_seq, 'act': act_seq, 'ce':ce})
        holder_eval.append(tmp)
        # Empty cache
        del lbls_batch, imgs_batch
        torch.cuda.empty_cache()
    df_eval = pd.concat(holder_eval).reset_index(None, True).assign(tt=lambda x: np.where(x.ids.isin(idt_val), 'Validation', 'Training'))
    r2_eval = df_eval.groupby(['tt']).apply(lambda x: r2_score(x.act, x.pred)).reset_index().rename(columns={0:'val'}).assign(metric='r2')
    ce_eval = df_eval.groupby(['tt']).ce.mean().reset_index().rename(columns={'ce':'val'}).assign(metric='ce')
    perf_eval = pd.concat([r2_eval, ce_eval]).assign(batch='eval')
    perf_train = pd.DataFrame({'tt': np.repeat(['Training'], 2), 'val': [r2_train, ce_train], 'batch': 'train', 'metric':['r2','ce']})
    perf_ee = pd.concat([perf_eval, perf_train]).assign(epoch=ee+1)
    epoch_loss.append(perf_ee)
    print(perf_ee)

    # Get run-time
    tdiff = time() - tnow
    print('Epoch took %i seconds, ETA: %i seconds' % (tdiff, (nepoch-ee-1)*tdiff) )
    tnow = time()

    if (ee + 1) % epoch_check == 0:
        print('------------ SAVING MODEL AT CHECKPOINT --------------')
        df_eval.to_csv(os.path.join(dir_ee,'df_'+str(ee+1)+'.csv'),index=False)
        torch.save(mdl.state_dict(), os.path.join(dir_ee,'mdl_'+str(ee+1)+'.pt'))
        yl = [0, df_cells.num_cell.max() + 1]
        tit = 'Estimed number of cells at epoch %i' % (ee + 1)
        tit = tit + '\n' + '\n'.join(r2_eval.apply(lambda x: x['tt'] + '=' + '{:0.3f}'.format(x['val']), 1))
        gg_scatter = (ggplot(df_eval,aes(x='pred',y='act',color='tt')) + theme_bw() +
                      geom_point() + geom_abline(intercept=0,slope=1,color='black',linetype='--') +
                      scale_y_continuous(limits=yl) + scale_x_continuous(limits=yl) +
                      ggtitle(tit) + labs(x='Predicted',y='Actual') +
                      facet_wrap('~tt') + guides(color=False))
        gg_scatter.save(os.path.join(dir_ee, 'cell_est.png'),width=8,height=4.5)

# SAVE LOSS AND NETWORK PLEASE!!
df_loss = pd.concat(epoch_loss).reset_index(None,True)
df_loss.to_csv(os.path.join(dir_hp,'mdl_performance.csv'),index=False)

# Make plots
cn_gg = ['metric','tt','batch']
tmp = df_loss.groupby(cn_gg).val.apply(lambda x:
 x.rolling(window=10,center=True).mean().fillna(method='bfill').fillna(method='ffill'))
df_loss.insert(df_loss.shape[1],'trend',tmp)
# Find minumum value
tmp = df_loss.assign(trend2=lambda x: np.where(x.metric=='r2',-x.trend,x.trend)).groupby(cn_gg).trend2.idxmin()

df_best = df_loss.loc[tmp.values]
df_best = df_best[df_best.tt=='Validation'].reset_index(None,True)

gg_loss = (ggplot(df_loss, aes(x='epoch',y='val',color='batch')) +
           geom_point(size=0.5) + theme_bw() +
           ggtitle('Performance over epochs') +
           facet_grid('metric~tt',scales='free_y') +
           theme(subplots_adjust={'wspace': 0.1}) +
           geom_line(aes(x='epoch',y='trend',color='batch')) +
           geom_vline(aes(xintercept='epoch'),data=df_best) +
           geom_text(aes(x='epoch+1',y='trend',label='epoch'),data=df_best,inherit_aes=False) +
           scale_y_continuous(limits=[0,df_loss.val.max()]))
gg_loss.save(os.path.join(dir_hp, 'performance_over_epochs.png'),width=12,height=6)

# Save the hyperparameter meta-data
df_slice.to_csv(os.path.join(dir_hp,'hyperparameters.csv'),index=False)
