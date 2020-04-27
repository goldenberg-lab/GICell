# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-np', '--num_param', type=int, default=8,
#                     help='First layer num_params, max=np*(2**4)')
# args = parser.parse_args()
# num_param = args.num_param
# print('Number of params: %i' % (num_param))
# import sys; sys.exit('leaving')
num_param=8
epoch_start = 250

import os, pickle
import numpy as np
import pandas as pd
from support_funs_GI import stopifnot, torch2array, sigmoid, comp_plt, sumax3, meanax3
from time import time
import torch
from unet_model import UNet
from sklearn import metrics

from helper_torch import CellCounterDataset, img2tensor, randomRotate, randomFlip
from torchvision import transforms
from torch.utils import data

import matplotlib

if not matplotlib.get_backend().lower() == 'agg':
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

######################################
## --- (1) PREP DATA AND MODELS --- ##

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_output, 'figures')
lst_dir = [dir_output, dir_figures]
[stopifnot(z) for z in lst_dir]
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
if not os.path.exists(dir_checkpoint):
    print('making checkpoint folder')
    os.mkdir(dir_checkpoint)

# Load data
di_img_point = pickle.load(open(os.path.join(dir_output, 'di_img_point.pickle'), 'rb'))
ids_tissue = list(di_img_point.keys())
valid_cells = ['eosinophil', 'neutrophil', 'plasma', 'enterocyte', 'other', 'lymphocyte']
# Sanity check
for idt in ids_tissue:
    tot1 = np.round(di_img_point[idt]['lbls'].sum() / 9).astype(int)
    tot2 = di_img_point[idt]['pts'].shape[0]
    df1 = dict(zip(valid_cells, [np.round(di_img_point[idt]['lbls'][:, :, k].sum() / 9, 0).astype(int) for k
                                 in range(len(valid_cells))]))
    df1 = pd.DataFrame(df1.items(), columns=['cell', 'n'])
    df2 = di_img_point[ids_tissue[0]]['pts'].cell.value_counts().reset_index().rename(
        columns={'cell': 'n', 'index': 'cell'})
    checka = tot1 == tot2
    checkb = df2.merge(df1, 'left', 'cell').assign(iseq=lambda x: x.n_x == x.n_y).iseq.all()
    if not checka and checkb:
        print('ID: %s\nTotal pixels: %i, total points: %i\n%s' % (idt, tot1, tot2, df2.merge(df1, 'left', 'cell')))
        stopifnot(False)

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device('cuda' if use_cuda else 'cpu')

# Load the model
pfac = 9  # Number of cells have been multiplied by 9 (basically)
multiclass = True
num_classes = 6
torch.manual_seed(1234)
mdl = UNet(n_channels=3, n_classes=num_classes, bl=num_param)
mdl.to(device)
if epoch_start > 0:
    fn_cp = pd.Series(os.listdir(dir_checkpoint))
    fn_cp = fn_cp[fn_cp.str.contains('epoch_'+str(epoch_start))]
    if len(fn_cp) == 1:
        path = os.path.join(dir_checkpoint, 'epoch_' + str(epoch_start), 'mdl_'+str(epoch_start)+'.pt')
    mdl.load_state_dict(torch.load(path))
else:
    # Will output bias with matching log-odds
    mat = np.vstack([sumax3(di_img_point[z]['lbls']) for z in di_img_point]).astype(int)
    mu_pixels = mat.mean(0) / (501 ** 2)
    b0 = np.log(mu_pixels / (1 - mu_pixels)).astype(np.float32)
    print('Setting bias to: %s,' % dict(zip(valid_cells, np.round(b0, 2))))
    with torch.no_grad():
        [mdl.outc.conv.bias[k].fill_(b0[k]) for k in range(len(b0))]

# Check CUDA status for model
print('Are network parameters cuda?: %s' %
      all([z.is_cuda for z in mdl.parameters()]))

# Calculate the weights
w0 = 1 / (np.exp(b0) / np.sum(np.exp(b0)))
w0 = w0 / w0.min()
# pd.DataFrame({'cell':valid_cells,'weight':w0})
# pd.concat([di_img_point[idt]['pts'].cell for idt in ids_tissue]).value_counts(True)

# Optimizer
optimizer = torch.optim.Adagrad(params=mdl.parameters(), lr=1e-4)

tnow = time()
# Loop through images and make sure they average number of cells equal
mat = np.zeros([len(ids_tissue), num_classes])
for ii, idt in enumerate(ids_tissue):
    print('ID-tissue %s (%i of %i)' % (idt, ii + 1, len(ids_tissue)))
    tens = img2tensor(device)([di_img_point[idt]['img'], di_img_point[idt]['lbls']])[0]
    tens = tens.reshape([1, 3, 501, 501])
    arr = torch2array(tens)
    with torch.no_grad():
        logits = mdl(tens)
        ncl = meanax3(logits.cpu().mean(0).T)
        mat[ii] = ncl
# print(mat.mean(axis=0)); print(mu_pixels*501**2); print(mu_cells); print(b0)
print('Forward pass took %i seconds' % (time() - tnow))
torch.cuda.empty_cache()
print(pd.DataFrame({'b0': b0, 'forward': mat.mean(0)}))

################################
## --- (2) BEGIN TRAINING --- ##

# Calculate eosinophilic ratio for each
df_eosophils = pd.concat(
    [di_img_point[idt]['pts'].cell.value_counts().reset_index().assign(id=idt) for idt in ids_tissue])
df_eosophils = df_eosophils.pivot('id', 'index', 'cell').fillna(0).reset_index()
cn_den = ['eosinophil', 'neutrophil', 'plasma', 'lymphocyte']
eratio = df_eosophils.eosinophil / (df_eosophils[cn_den].sum(1) + 1)
df_ratio = pd.DataFrame({'id': df_eosophils.id, 'ratio': eratio}).sort_values('ratio').reset_index(None, True)
quantiles = [0.25, 0.50, 0.75]
idt_val = []
for qq in quantiles:
    idx = ((df_ratio.ratio - df_ratio.ratio.quantile(qq)) ** 2).idxmin()
    idt_val.append(df_ratio.loc[idx, 'id'])
    print(df_ratio.iloc[[idx]])

# Select instances for training/validation
q1 = pd.Series(ids_tissue)
idt_train = q1[~q1.isin(idt_val)].to_list()
print('%i training samples\n%i validation samples: %s' %
      (len(idt_train), len(idt_val), ', '.join(idt_val)))

# Create datasetloader class
train_params = {'batch_size': 2, 'shuffle': True}
val_params = {'batch_size': len(idt_val), 'shuffle': False}
eval_params = {'batch_size': 1, 'shuffle': False}

train_transform = transforms.Compose([randomRotate(tol=1e-4), randomFlip(), img2tensor(device)])
train_data = CellCounterDataset(di=di_img_point, ids=idt_train, transform=train_transform, multiclass=multiclass)
train_gen = data.DataLoader(dataset=train_data, **train_params)
val_transform = transforms.Compose([img2tensor(device)])
val_data = CellCounterDataset(di=di_img_point, ids=idt_val, transform=val_transform, multiclass=multiclass)
val_gen = data.DataLoader(dataset=val_data, **val_params)
eval_data = CellCounterDataset(di=di_img_point, ids=idt_train + idt_val, transform=val_transform, multiclass=multiclass)
eval_gen = data.DataLoader(dataset=eval_data, **eval_params)

tnow = time()
nepochs = 5000
df_loss = pd.DataFrame({'ce_train': np.zeros(nepochs-epoch_start),
                        'ce_val': np.zeros(nepochs-epoch_start)})
df_rsq = []
ee, ii = epoch_start, 1
epoch_iter = np.arange(epoch_start, nepochs)
for ee in epoch_iter:
    print('--------- EPOCH %i of %i ----------' % (ee + 1, nepochs))
    ii = 0
    np.random.seed(ee)
    torch.manual_seed(ee)
    lst_ce, lst_comp = [], []
    for ids_batch, lbls_batch, imgs_batch in train_gen:
        ii += 1
        ids_batch = list(ids_batch)
        nbatch = len(ids_batch)
        print('-- batch %i of %i: %s --' % (ii, len(train_gen), ', '.join(ids_batch)))
        # --- Forward pass --- #
        optimizer.zero_grad()
        logits = mdl(imgs_batch)
        assert logits.shape == lbls_batch.shape
        # Weight the less frquent class
        wmat = np.stack([np.zeros([nbatch] + list(lbls_batch.shape[2:])) + w0[k] for k in range(len(w0))], 1)
        wmat = torch.tensor(wmat.astype(np.float32)).to(device)  # wmat = wmat*0+1
        # Binary loss
        criterion = torch.nn.BCEWithLogitsLoss(weight=wmat, reduction='mean')
        loss = criterion(input=logits, target=lbls_batch)
        # --- Backward pass --- #
        loss.backward()
        # --- Gradient step --- #
        optimizer.step()
        torch.cuda.empty_cache()  # Empty cache
        # --- Performance --- #
        ii_loss = loss.cpu().detach().numpy() + 0
        ii_phat = sigmoid(logits.cpu().detach().numpy())
        ii_pred = pd.DataFrame({'cell': valid_cells, 'n': sumax3(ii_phat.sum(0).transpose(1, 2, 0)) / pfac})
        ii_act = pd.concat([di_img_point[idt]['pts'].cell for idt in ids_batch]).value_counts().reset_index()
        ii_act.rename(columns={'cell': 'n', 'index': 'cell'}, inplace=True)
        ii_act = ii_act.merge(ii_pred, 'outer', 'cell', suffixes=('_act', '_pred')).fillna(0)
        lst_ce.append(ii_loss)
        lst_comp.append(ii_act)
        print('Cross-entropy loss: %0.4f\nPredicted: %s, actual: %s' %
              (ii_loss, ', '.join(ii_act.n_pred.astype(int).astype(str)),
               ', '.join(ii_act.n_act.astype(int).astype(str))))
    ii_comp = pd.concat(lst_comp)
    rho_train = ii_comp.groupby('cell').apply(lambda x: metrics.r2_score(x.n_act, x.n_pred)).reset_index()
    rho_train.rename(columns={0: 'rsq'}, inplace=True)
    ce_train = np.mean(lst_ce)
    # Evaluate model on validation data
    with torch.no_grad():
        for ids_batch, lbls_batch, imgs_batch in val_gen:
            nbatch = len(ids_batch)
            logits = mdl.eval()(imgs_batch)
            ii_phat = sigmoid(logits.cpu().detach().numpy())
            wmat = np.stack([np.zeros([nbatch] + list(lbls_batch.shape[2:])) + w0[k] for k in range(len(w0))], 1)
            wmat = torch.tensor(wmat.astype(np.float32)).to(device)
            criterion = torch.nn.BCEWithLogitsLoss(weight=wmat, reduction='mean')
            ce_val = criterion(input=logits, target=lbls_batch).cpu().detach().numpy() + 0
    torch.cuda.empty_cache()  # Empty cache
    print('Cross-entropy - training: %0.4f, validation: %0.4f\n\ncell-count corr:\n%s' %
          (ce_train, ce_val, rho_train))
    df_loss.iloc[ee] = [ce_train, ce_val]
    df_rsq.append(rho_train.assign(epoch=ee + 1))
    print('Epoch took %i seconds' % int(time() - tnow))
    tnow = time()
    # Save plots and network every 250 epochs
    if (ee + 1) % 250 == 0:
        print('------------ SAVING MODEL AT CHECKPOINT --------------')
        dir_ee = os.path.join(dir_checkpoint, 'epoch_' + str(ee + 1))
        if not os.path.exists(dir_ee):
            os.mkdir(dir_ee)
        with torch.no_grad():
            holder = []
            for ids_batch, lbls_batch, imgs_batch in eval_gen:
                id = ids_batch[0]
                print('Making image for: %s' % id)
                logits = mdl.eval()(imgs_batch).cpu().detach().numpy()
                logits = sigmoid(logits[0, :, :, :].transpose(1, 2, 0))
                img = torch2array(imgs_batch)[:, :, :, 0]
                gaussian = di_img_point[id]['lbls']
                # comp_plt(arr=img, pts=logits, gt=gaussian, path=dir_ee, fn=id + '.png',
                #          thresh=sigmoid(np.floor(b0)), lbls=valid_cells)
                tmp = pd.DataFrame({'id':id, 'cell':valid_cells,
                                    'act':sumax3(gaussian) / pfac,'pred':sumax3(logits) / pfac})
                holder.append(tmp)
        df_ee = pd.concat(holder).reset_index(None,True)
        df_ee['tt'] = np.where(df_ee.id.isin(idt_val), 'Validation', 'Training')
        df_ee.to_csv(os.path.join(dir_ee, 'df_' + str(ee + 1) + '.csv'), index=False)
        rho_all = df_ee.groupby('cell').apply(lambda x: metrics.r2_score(x.act, x.pred)).reset_index()
        rho_all.rename(columns={0: 'rsq'}, inplace=True)
        # --- make figure --- #
        plt.close()
        g = sns.FacetGrid(df_ee, hue='tt', col='cell',col_wrap=3,
                          sharex=False, sharey=False)  #height=5, aspect=1.2
        g.map(plt.scatter, 'pred', 'act')
        g.set_xlabels('Predicted')
        g.set_ylabels('Actual')
        g.fig.suptitle('Estimed number of cells at epoch %i' % (ee + 1))
        g.fig.subplots_adjust(top=0.85)
        g.add_legend()
        g._legend.set_title('')
        for ax in g.axes.flatten():
            xmx = int(max(max(ax.get_xlim()), max(ax.get_ylim()))) + 1
            xmi = int(min(min(ax.get_xlim()), min(ax.get_ylim()))) - 1
            ax.set_ylim((xmi, xmx))
            ax.set_xlim((xmi, xmx))
            ax.plot([xmi, xmx], [xmi, xmx], '--', c='black')
            cell = ax.get_title().replace('cell = ','')
            rsq = rho_all.loc[rho_all.cell == cell,'rsq'].to_list()[0]
            tt = 'cell: %s , rsq: %0.3f' % (cell, rsq)
            ax.set_title(tt)
        g.fig.savefig(os.path.join(dir_ee, 'cell_est.png'))
        # ---- #
        torch.cuda.empty_cache()  # Empty cache
        # Save network
        torch.save(mdl.state_dict(), os.path.join(dir_ee, 'mdl_' + str(ee + 1) + '.pt'))

# SAVE LOSS AND NETWORK PLEASE!!
df_loss.insert(0, 'epoch', np.arange(nepochs) + 1)
df_loss = df_loss[df_loss.train != 0].reset_index(None, True)
df_loss.to_csv(os.path.join(dir_output, 'mdl_performance.csv'), index=False)
