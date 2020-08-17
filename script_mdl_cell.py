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

# # for beta testing  ['eosinophil']
# cells, num_epochs, batch_size, learning_rate, num_params, epoch_check = ['eosinophil'], 250, 2, 1e-3, 32, 1
valid_cells = ['eosinophil', 'neutrophil', 'plasma', 'enterocyte', 'other', 'lymphocyte']
assert all([z in valid_cells for z in cells])

print('Cells: %s\nnum_epochs: %i\nbatch_size: %i\nlearning_rate: %0.3f, num_params: %i' % (', '.join(cells), num_epochs, batch_size, learning_rate, num_params))
# import sys
# sys.exit('end of script')

import os
import pickle
import numpy as np
import pandas as pd
from funs_support import stopifnot, torch2array, sigmoid, comp_plt, makeifnot, t2n
from time import time
import torch
from funs_unet import UNet
from sklearn import metrics

from funs_torch import CellCounterDataset, img2tensor, randomRotate, randomFlip
from torchvision import transforms
from torch.utils import data

import matplotlib
if not matplotlib.get_backend().lower() == 'agg':
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device('cuda' if use_cuda else 'cpu')

######################################
## --- (1) PREP DATA AND MODELS --- ##

from datetime import datetime
# Get current day
dnow = datetime.now().strftime('%Y_%m_%d')

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_output, 'figures')
lst_dir = [dir_output, dir_figures]
[stopifnot(z) for z in lst_dir]
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
dir_cell = os.path.join(dir_checkpoint, '_'.join(np.sort(cells)))
dir_datecell = os.path.join(dir_cell, dnow)
makeifnot(dir_checkpoint)
makeifnot(dir_cell)
makeifnot(dir_datecell)

# Load data
di_img_point = pickle.load(open(os.path.join(dir_output, 'di_img_point.pickle'), 'rb'))
ids_tissue = list(di_img_point.keys())
count1, count2 = np.round(di_img_point[ids_tissue[0]]['lbls'].sum() / 9).astype(int), \
                    di_img_point[ids_tissue[0]]['pts'].shape[0]
assert count1 == count2

# Change pixel count to match the cell types
idx_cell = np.where(pd.Series(valid_cells).isin(cells))[0]
for idt in ids_tissue:
    tmp = di_img_point[idt]['lbls'].copy()
    tmp2 = np.atleast_3d(tmp[:, :, idx_cell].sum(2))
    tmp3 = di_img_point[idt]['pts'].copy()
    tmp3 = tmp3[tmp3.cell.isin(cells)]
    di_img_point[idt]['lbls'] = tmp2
    assert np.abs( tmp3.shape[0] - (tmp2.sum() / 9) ) < 1
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
mdl = UNet(n_channels=3, n_classes=1, bl=num_params)
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
# Original list from May
idt_val1 = ['R9I7FYRB_Transverse_17', 'RADS40DE_Rectum_13', '8HDFP8K2_Transverse_5',
           '49TJHRED_Descending_46', 'BLROH2RX_Cecum_72', '8ZYY45X6_Sigmoid_19',
           '6EAWUIY4_Rectum_56', 'BCN3OLB3_Descending_79']
# Double validation list for August samples
idt_val2 = ['ESZOXUA8_Transverse_80', '49TJHRED_Rectum_30', '8HDFP8K2_Ascending_35',
            'BCN3OLB3_Descending_51', 'ESZOXUA8_Descending_91', '8ZYY45X6_Ascending_68',
            'E9T0C977_Sigmoid_34', '9U0ZXCBZ_Cecum_41']
assert len(np.intersect1d(idt_val1, idt_val2))==0
idt_val = idt_val1 + idt_val2

idt_train = df_cells.id[~df_cells.id.isin(idt_val)].to_list()
print('%i training samples\n%i validation samples' % (len(idt_train), len(idt_val)))

# Create datasetloader class
train_params = {'batch_size': batch_size, 'shuffle': True}
val_params = {'batch_size': len(idt_val),'shuffle': False}
eval_params = {'batch_size': 1,'shuffle': False}

multiclass = False

train_transform = transforms.Compose([randomRotate(tol=1e-4), randomFlip(), img2tensor(device)])
train_data = CellCounterDataset(di=di_img_point, ids=idt_train, transform=train_transform,
                                multiclass=multiclass)
train_gen = data.DataLoader(dataset=train_data,**train_params)
val_transform = transforms.Compose([img2tensor(device)])
val_data = CellCounterDataset(di=di_img_point, ids=idt_val, transform=val_transform,
                              multiclass=multiclass)
val_gen = data.DataLoader(dataset=val_data,**val_params)
eval_data = CellCounterDataset(di=di_img_point, ids=idt_train + idt_val, transform=val_transform, multiclass=multiclass)
eval_gen = data.DataLoader(dataset=eval_data, **eval_params)

tnow = time()
mat_loss = np.zeros([num_epochs, 4])
ee, ii = 0, 1
for ee in range(num_epochs):
    print('--------- EPOCH %i of %i ----------' % (ee+1, num_epochs))
    mdl.train()
    ii = 0
    np.random.seed(ee)
    torch.manual_seed(ee)
    lst_ce, lst_pred_act = [], []
    for ids_batch, lbls_batch, imgs_batch in train_gen:
        ii += 1
        ids_batch = list(ids_batch)
        nbatch = len(ids_batch)
        print('-- batch %i of %i: %s --' % (ii, len(train_gen), ', '.join(ids_batch)))
        # --- Forward pass --- #
        optimizer.zero_grad()
        logits = mdl(imgs_batch)
        assert logits.shape == lbls_batch.shape
        loss = criterion(input=logits,target=lbls_batch)
        # --- Backward pass --- #
        loss.backward()
        # --- Gradient step --- #
        optimizer.step()
        torch.cuda.empty_cache()  # Empty cache
        # --- Performance --- #
        ii_loss = t2n(loss)+0
        ii_phat = sigmoid(t2n(logits))
        ii_pred_act = np.zeros([nbatch, 2])
        for kk in range(nbatch):
            ii_pred_act[kk, 0] = ii_phat[kk].sum() / pfac
            ii_pred_act[kk, 1] = di_img_point[ids_batch[kk]]['lbls'].sum() / pfac
        lst_ce.append(ii_loss)
        lst_pred_act.append(ii_pred_act)
        print('Cross-entropy loss: %0.4f' % ii_loss)
    mat_pred_act = pd.DataFrame(np.vstack(lst_pred_act),columns=['pred','act'])
    rho_train = metrics.r2_score(mat_pred_act.act, mat_pred_act.pred)
    ce_train = np.mean(lst_ce)
    torch.cuda.empty_cache()  # Empty cache
    # Evaluate model on validation data
    mdl.eval()
    with torch.no_grad():
        for ids_batch, lbls_batch, imgs_batch in val_gen:
            logits = mdl(imgs_batch)
            torch.cuda.empty_cache()  # Empty cache
            ii_phat = sigmoid(t2n(logits))
            ce_val = t2n(criterion(input=logits,target=lbls_batch))+0
            nbatch = len(ids_batch)
            ii_pred_act = np.zeros([nbatch,2])
            for kk in range(nbatch):
                ii_pred_act[kk, 0] = ii_phat[kk].sum() / pfac
                ii_pred_act[kk, 1] = di_img_point[ids_batch[kk]]['lbls'].sum() / pfac
                print(ids_batch); print(ii_pred_act[kk])
    val_pa = pd.DataFrame(ii_pred_act,columns=['pred','act'])
    val_pa.insert(0,'id',ids_batch)
    rho_val = metrics.r2_score(val_pa.act, val_pa.pred)
    print(np.round(val_pa,1))
    torch.cuda.empty_cache()  # Empty cache
    # Print performance
    print('Cross-entropy - training: %0.4f, validation: %0.4f'
          'R-squared - training: %0.3f, validation: %0.3f' %
          (ce_train, ce_val, rho_train, rho_val))
    mat_loss[ee] = [ce_train, ce_val, rho_train, rho_val]
    tdiff = time() - tnow
    print('Epoch took %i seconds, ETA: %i seconds' % (tdiff, (num_epochs-ee-1)*tdiff) )
    tnow = time()
    # Save plots and network every X epochs
    if (ee+1) % epoch_check == 0:
        print('------------ SAVING MODEL AT CHECKPOINT --------------')
        dir_ee = os.path.join(dir_datecell,'epoch_'+str(ee+1))
        if not os.path.exists(dir_ee):
            os.mkdir(dir_ee)
        mdl.eval()
        with torch.no_grad():
            holder = []
            for ids_batch, lbls_batch, imgs_batch in eval_gen:
                id = ids_batch[0]
                print('Making image for: %s' % id)
                logits = mdl.eval(imgs_batch)
                logits = logits.cpu().detach().numpy().sum(0).transpose(1,2,0)
                torch.cuda.empty_cache()  # Empty cache
                phat = sigmoid(logits)
                img = torch2array(imgs_batch).sum(3)
                gaussian = di_img_point[id]['lbls'].copy()
                if isinstance(b0,float):
                    thresher = sigmoid(np.floor(np.array([b0])))
                else:
                    thresher = sigmoid(np.floor(b0))
                tt = 'train'
                if id in idt_val:
                    tt = 'valid'
                comp_plt(arr=img,pts=phat,gt=gaussian,path=dir_ee,fn=tt+'_'+id+'.png',
                         lbls=[', '.join(cells)], thresh=thresher)
                holder.append([id, gaussian.sum() / pfac, phat.sum() / pfac])
        df_ee = pd.DataFrame(holder,columns=['id','act','pred'])
        df_ee['tt'] = np.where(df_ee.id.isin(idt_val), 'Validation', 'Training')
        df_ee.to_csv(os.path.join(dir_ee,'df_'+str(ee+1)+'.csv'),index=False)
        r2_ee = metrics.r2_score(df_ee.act, df_ee.pred)
        # --- make figure --- #
        plt.close()
        g = sns.FacetGrid(df_ee,hue='tt',height=5,aspect=1.2)
        g.map(plt.scatter,'pred','act')
        g.set_xlabels('Predicted')
        g.set_ylabels('Actual')
        g.fig.suptitle('Estimed number of cells at epoch %i\nR-squared: %0.3f' % (ee + 1, r2_ee))
        g.fig.subplots_adjust(top=0.85)
        g.add_legend()
        g._legend.set_title('')
        for ax in g.axes.flatten():
            xmx = int(max(max(ax.get_xlim()),max(ax.get_ylim())))+1
            xmi = int(min(min(ax.get_xlim()),min(ax.get_ylim())))-1
            ax.set_ylim((xmi, xmx))
            ax.set_xlim((xmi, xmx))
            ax.plot([xmi, xmx], [xmi, xmx], '--',c='black')
        g.fig.savefig(os.path.join(dir_ee, 'cell_est.png'))
        # ---- #
        torch.cuda.empty_cache()  # Empty cache
        # Save network
        torch.save(mdl.state_dict(), os.path.join(dir_ee,'mdl_'+str(ee+1)+'.pt'))

# SAVE LOSS AND NETWORK PLEASE!!
df_loss = pd.DataFrame(mat_loss,columns=['ce_train','ce_val','r2_train','r2_val'])
df_loss.insert(0,'epoch',np.arange(num_epochs)+1)
df_loss = df_loss[df_loss.ce_train != 0].reset_index(None,True)
df_loss.to_csv(os.path.join(dir_cell,'mdl_performance.csv'),index=False)
