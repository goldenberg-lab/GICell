import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cells', type=str, help='List of comma-seperated cell types',
                    default='eosinophil,neutrophil,plasma,enterocyte,other,lymphocyte')
parser.add_argument('-ne', '--num_epochs', type=int, default=1000, help='Number of epochs')
parser.add_argument('-bs', '--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('-np', '--num_params', type=int, default=8, help='Number of initial params for NNet')
parser.add_argument('-ec', '--epoch_check', type=int, default=250, help='Iteration number to save checkpoint')

args = parser.parse_args()
cells = args.cells.split(',')
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
num_params = args.num_params
epoch_check = args.epoch_check

# # for beta testing
# cells = ['eosinophil','neutrophil','plasma','lymphocyte']
# learning_rate, num_params = 0.001, 16
# num_epochs, epoch_check, batch_size = 2, 1, 2

valid_cells = ['eosinophil', 'neutrophil', 'plasma', 'enterocyte', 'other', 'lymphocyte']
assert all([z in valid_cells for z in cells])

print('Cells: %s\nnum_epochs: %i\nbatch_size: %i\nlearning_rate: %0.3f, num_params: %i' % (', '.join(cells), num_epochs, batch_size, learning_rate, num_params))
# import sys
# sys.exit('end of script')

import os
import pickle
import numpy as np
import pandas as pd
from funs_support import stopifnot, torch2array, sigmoid, comp_plt, makeifnot, t2n, find_dir_cell
from time import time
import torch
from funs_unet import UNet
from sklearn.metrics import r2_score
from datetime import datetime

from funs_torch import CellCounterDataset, img2tensor, randomRotate, randomFlip
from torchvision import transforms
from torch.utils import data

import matplotlib
if not matplotlib.get_backend().lower() == 'agg':
    matplotlib.use('Agg')
from plotnine import *

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device('cuda' if use_cuda else 'cpu')

######################################
## --- (1) PREP DATA AND MODELS --- ##

# Hyperparameter configs
df_slice = pd.DataFrame({'lr':learning_rate, 'num_params':num_params,
                         'num_epochs':num_epochs,'epoch_check':epoch_check,
                         'batch_size':batch_size},index=[0])

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
makeifnot(dir_checkpoint)
makeifnot(dir_cell)
makeifnot(dir_datecell)
makeifnot(dir_hp)

# Load data
di_img_point = pickle.load(open(os.path.join(dir_output, 'di_img_point.pickle'), 'rb'))
ids_tissue = list(di_img_point.keys())
count1, count2 = np.round(di_img_point[ids_tissue[0]]['lbls'].sum() / 9).astype(int), \
                    di_img_point[ids_tissue[0]]['pts'].shape[0]
assert np.abs(count1/count2-1) < 0.02

# Change pixel count to match the cell types
idx_cell = np.where(pd.Series(valid_cells).isin(cells))[0]
holder = []
for kk, idt in enumerate(ids_tissue):
    tmp = di_img_point[idt]['lbls'].copy()
    tmp2 = np.atleast_3d(tmp[:, :, idx_cell].sum(2))
    tmp3 = di_img_point[idt]['pts'].copy()
    tmp3 = tmp3[tmp3.cell.isin(cells)]
    gt, est = tmp3.shape[0], (tmp2.sum() / 9)
    if gt > 0:
        err_pct = gt / est - 1
        assert np.abs(err_pct) < 0.02
    di_img_point[idt]['lbls'] = tmp2
    holder.append(err_pct)
    del tmp, tmp2, tmp3

# Get the mean number of cells
pfac = 9  # Number of cells have been multiplied by 9 (basically)
mu_pixels = np.mean([di_img_point[z]['lbls'].mean() for z in di_img_point])
mu_cells = pd.concat([di_img_point[z]['pts'].cell.value_counts().reset_index().rename(columns={'index':'cell','cell':'n'}) for z in di_img_point])
mu_cells = mu_cells[mu_cells.cell.isin(cells)].n.sum() / len(di_img_point)
print('Error: %0.1f' % (mu_pixels * 501**2 / pfac - mu_cells))
b0 = np.log(mu_pixels / (1 - mu_pixels))

# Calculate distribution of cells types across images
df_cells = pd.concat([di_img_point[idt]['pts'].cell.value_counts().reset_index().assign(id=idt) for idt in ids_tissue])
df_cells = df_cells.pivot('id', 'index', 'cell').fillna(0).reset_index()
num_cell = df_cells[cells].sum(1).astype(int)
num_agg = df_cells.drop(columns=['id']).sum(1).astype(int)
df_cells = pd.DataFrame({'id':df_cells.id,'num_cell':num_cell,'num_agg':num_agg})
df_cells = df_cells.assign(ratio = lambda x: x.num_cell/x.num_agg)
df_cells = df_cells.sort_values('ratio').reset_index(None,True)
# tmp = df_cells.copy()
# num_val = int(np.floor(df_cells.shape[0] * 0.2))
# quants = np.linspace(0,1,num_val+2)[1:-1]
# cc_find = df_cells.ratio.quantile(quants,interpolation='lower').values
# holder = []
# for cc in cc_find:
#     idt = tmp.loc[tmp.ratio == cc,'id'].head(1).to_list()[0]
#     holder.append(idt)
#     tmp = tmp[tmp.id != idt]
# df_cells.insert(1,'tt',np.where(df_cells.id.isin(holder),'test','train'))
print(df_cells)

# Load the model
torch.manual_seed(1234)
mdl = UNet(n_channels=3, n_classes=1, bl=num_params, batchnorm=True)
mdl.to(device)
# mdl.load_state_dict(torch.load(os.path.join(dir_ee,'mdl_1000.pt')))
with torch.no_grad():
    mdl.outc.conv.bias.fill_(b0)
# Check CUDA status for model
print('Are network parameters cuda?: %s' %
      all([z.is_cuda for z in mdl.parameters()]))

# Binary loss
criterion = torch.nn.BCEWithLogitsLoss()
# Optimizer
optimizer = torch.optim.Adagrad(params=mdl.parameters(), lr=learning_rate)

tnow = time()
# Loop through images and make sure they average number of cells equal
mat = np.zeros([len(ids_tissue), 2])
for ii, idt in enumerate(ids_tissue):
    print('ID-tissue %s (%i of %i)' % (idt, ii + 1, len(ids_tissue)))
    tens = img2tensor(device)([di_img_point[idt]['img'],di_img_point[idt]['lbls']])[0]
    tens = tens.reshape([1,3,501,501]) / 255
    arr = torch2array(tens)
    with torch.no_grad():
        logits = mdl(tens)
        ncl = logits.cpu().mean().numpy()+0
        nc = torch.sigmoid(logits).cpu().sum().numpy()+0
        mat[ii] = [nc, ncl]
print('Script took %i seconds' % (time() - tnow))
print(mat.mean(axis=0)); print(mu_pixels*501**2); print(mu_cells); print(b0)
torch.cuda.empty_cache()

################################
## --- (2) BEGIN TRAINING --- ##

# Select instances for training/validation
# Original list from May (8 samples)
idt_val1 = ['R9I7FYRB_Transverse_17', 'RADS40DE_Rectum_13', '8HDFP8K2_Transverse_5',
           '49TJHRED_Descending_46', 'BLROH2RX_Cecum_72', '8ZYY45X6_Sigmoid_19',
           '6EAWUIY4_Rectum_56', 'BCN3OLB3_Descending_79']
# Double validation list for August samples (8 samples)
idt_val2 = ['ESZOXUA8_Transverse_80', '49TJHRED_Rectum_30', '8HDFP8K2_Ascending_35',
            'BCN3OLB3_Descending_51', 'ESZOXUA8_Descending_91', '8ZYY45X6_Ascending_68',
            'E9T0C977_Sigmoid_34', '9U0ZXCBZ_Cecum_41']
# Doubling validation set (Jan-2021) with +16
idt_val3 = ['TRS8XIRT_Rectum_76', 'Y7CXU9SM_Sigmoid_69', 'RADS40DE_Ascending_8', '8HDFP8K2_Rectum_74.png-points.tsv', '49TJHRED_Transverse_19', 'A6TT1X9U_Transverse_87.png-points.tsv', 'Y7CXU9SM_Transverse_94', 'MM6IXZVW_Ascending_18', 'MARQQRM5_Descending_2', '1OE1DR6N_Transverse_24', 'QF0TMM7V_Sigmoid_95', '02FQJM8D_Transverse_52', 'BCN3OLB3_Rectum_20', 'PZUZFPUN_Ascending_0.png-points.tsv', 'A6TT1X9U_Sigmoid_19.png-points.tsv', 'BOD4MJT0_Sigmoid_18.png-points.tsv']
# Check no overlap
idt_lst = [idt_val1, idt_val2, idt_val3]
for ii in range(0,len(idt_lst)-1):
    for jj in range(ii+1, len(idt_lst)):
        assert len(np.intersect1d(idt_lst[ii], idt_lst[jj])) == 0
# Get final set
idt_val = sum(idt_lst, [])
idt_train = df_cells.id[~df_cells.id.isin(idt_val)].to_list()
print('%i training samples\n%i validation samples' % (len(idt_train), len(idt_val)))

# Create datasetloader class
train_params = {'batch_size': batch_size, 'shuffle': True}
val_params = {'batch_size': batch_size,'shuffle': False}
eval_params = {'batch_size': batch_size,'shuffle': True}

multiclass = False

# Traiing
train_transform = transforms.Compose([randomRotate(tol=1e-4), randomFlip(),
                                      img2tensor(device)])
train_data = CellCounterDataset(di=di_img_point, ids=idt_train, transform=train_transform,
                                multiclass=multiclass)
train_gen = data.DataLoader(dataset=train_data,**train_params)
# Validation
val_transform = transforms.Compose([img2tensor(device)])
val_data = CellCounterDataset(di=di_img_point, ids=idt_val, transform=val_transform,
                              multiclass=multiclass)
val_gen = data.DataLoader(dataset=val_data,**val_params)
# Eval (all sample)
eval_data = CellCounterDataset(di=di_img_point, ids=idt_train + idt_val,
                               transform=val_transform, multiclass=multiclass)
eval_gen = data.DataLoader(dataset=eval_data, **eval_params)


tnow = time()
# cn_loss = pd.Series(np.repeat(['train_train','train_eval','val_eval'],2))+'_'+pd.Series(np.tile(['r2','ce'],3))
# print(cn_loss)
# mat_loss = np.zeros([num_epochs, len(cn_loss)])
epoch_loss = []
ee, ii = 0, 1
for ee in range(num_epochs):
    print('--------- EPOCH %i of %i ----------' % (ee+1, num_epochs))

    ### --- MODEL TRAINING --- ###
    mdl.train()
    np.random.seed(ee)
    torch.manual_seed(ee)
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
        ii_phat = sigmoid(t2n(logits))
        # Empty cache
        del lbls_batch, imgs_batch
        torch.cuda.empty_cache()
        ii_pred_act = np.zeros([nbatch, 2])
        for kk in range(nbatch):
            ii_pred_act[kk, 0] = ii_phat[kk].sum() / pfac
            ii_pred_act[kk, 1] = di_img_point[ids_batch[kk]]['lbls'].sum() / pfac
        lst_ce.append(ii_loss)
        lst_pred_act.append(ii_pred_act)
        lst_ids.append(ids_batch)
        if (ii + 1) % 25 == 0:
            print('-- batch %i of %i: %s --' % (ii+1, len(train_gen), ', '.join(ids_batch)))
            print('Cross-entropy loss: %0.6f' % ii_loss)
    mat_pred_act = pd.DataFrame(np.vstack(lst_pred_act),columns=['pred','act'])
    mat_pred_act.insert(0, 'ids', np.concatenate(lst_ids))
    r2_train = r2_score(mat_pred_act.act, mat_pred_act.pred)
    corr_train = np.corrcoef(mat_pred_act.act, mat_pred_act.pred)[0,1]
    ce_train = np.mean(lst_ce)

    # Create epoch checkpoint folder
    if (ee+1) % epoch_check == 0:
        dir_ee = os.path.join(dir_hp,'epoch_'+str(ee+1))
        makeifnot(dir_ee)

    ### --- MODEL EVALUATION --- ###
    mdl.eval()
    holder_eval = []
    with torch.no_grad():
        for ids_batch, lbls_batch, imgs_batch in eval_gen:
            if (ii + 1) % 25 == 0:
                print('Prediction for: %s' % ', '.join(ids_batch))
            logits = t2n(mdl(imgs_batch))
            ce = t2n(criterion(input=mdl(imgs_batch), target=lbls_batch))
            logits = logits.transpose(2, 3, 1, 0)
            phat = sigmoid(logits)
            gaussian = np.stack([di_img_point[ids]['lbls'].copy() for ids in ids_batch], 3)
            ids_seq = list(ids_batch)
            pred_seq = phat.sum(0).sum(0).sum(0) / pfac
            act_seq = gaussian.sum(0).sum(0).sum(0) / pfac
            tmp = pd.DataFrame({'ids': ids_seq, 'pred': pred_seq, 'act': act_seq, 'ce':ce})
            holder_eval.append(tmp)
            # if (ee+1) % epoch_check == 0:
            #     print('Making image for: %s' % ', '.join(ids_batch))
            #     img = torch2array(imgs_batch)
            #     if isinstance(b0, float):
            #         thresher = sigmoid(np.floor(np.array([b0])))
            #     else:
            #         thresher = sigmoid(np.floor(b0))
            #     for j, ids in enumerate(ids_batch):
            #         img_j, phat_j, gt_j = img[:,:,:,j], phat[:,:,:,j], gaussian[:,:,:,j]
            #         tt = 'train'
            #         if ids in idt_val:
            #             tt = 'valid'
            #         comp_plt(arr=img_j,pts=phat_j,gt=gt_j, path=dir_ee, fn=tt+'_'+ids+'.png',
            #                  lbls=[', '.join(cells)], thresh=thresher)
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
    print('Epoch took %i seconds, ETA: %i seconds' % (tdiff, (num_epochs-ee-1)*tdiff) )
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
