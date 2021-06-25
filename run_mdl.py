import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--is_eosin', dest='is_eosin', action='store_true')
parser.add_argument('--is_inflam', dest='is_inflam', action='store_true')
parser.set_defaults(is_eosin=False, is_inflam=False)
parser.add_argument('--nepoch', type=int, default=1000, help='Number of epochs')
parser.add_argument('--batch', type=int, default=1, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--p', type=int, default=8, help='Number of initial params for NNet')
parser.add_argument('--nfill', type=int, default=1, help='How many points to pad around pixel annotation point')
args = parser.parse_args()
is_eosin, is_inflam = args.is_eosin, args.is_inflam
nepoch, batch, lr, p, nfill = args.nepoch, args.batch, args.lr, args.p, args.nfill

# # for beta testing
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

import os
import hickle
import gc
import numpy as np
import pandas as pd
from funs_support import torch2array, sigmoid, makeifnot, t2n, find_dir_cell
from time import time
import torch
from mdls.unet import UNet
from sklearn.metrics import r2_score
from datetime import datetime
from funs_torch import CellCounterDataset, img2tensor, randomRotate, randomFlip
from torchvision import transforms
from torch.utils import data

# import matplotlib
# if not matplotlib.get_backend().lower() == 'agg':
#     matplotlib.use('Agg')
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

######################################
## --- (1) PREP DATA AND MODELS --- ##

# --- (i) Load data --- #
path_pickle = os.path.join(dir_output, 'annot_hsk.pickle')
di_data = hickle.load(path_pickle)
gc.collect()
ids_tissue = list(di_data.keys())
count1 = np.round(di_data[ids_tissue[0]]['lbls'].sum() / 9).astype(int)
count2 = di_data[ids_tissue[0]]['pts'].shape[0]
assert np.abs(count1/count2-1) < 0.02

# --- (ii) Set labels to match cell type --- #
idx_cell = np.where(pd.Series(valid_cells).isin(cells))[0]
holder = []
for kk, idt in enumerate(ids_tissue):
    tmp = di_data[idt]['lbls'].copy()
    tmp2 = np.atleast_3d(tmp[:, :, idx_cell].sum(2))
    tmp3 = di_data[idt]['pts'].copy()
    tmp3 = tmp3[tmp3.cell.isin(cells)]
    est = tmp2.sum() / fillfac
    if len(tmp3) > 0:
        err_pct = len(tmp3) / est - 1
        assert np.abs(err_pct) < 0.02
    di_data[idt]['lbls'] = tmp2
    holder.append(err_pct)
    del tmp, tmp2, tmp3
dat_err_pct = pd.DataFrame({'idt':ids_tissue, 'err':holder}).assign(err=lambda x: x.err.abs())
dat_err_pct = dat_err_pct.sort_values('err',ascending=False).round(2).reset_index(None,True)
print(dat_err_pct.head())

# --- (iii) Check image/label size concordance --- #
dat_imsize = pd.DataFrame([di_data[idt]['lbls'].shape[:2] + di_data[idt]['img'].shape[:2] for idt in ids_tissue])
n_pixels = dat_imsize.iloc[0,0]
assert np.all(dat_imsize == n_pixels)

# --- (iv) Get mean number of cells/pixels for intercept initialization --- #
mu_pixels = np.mean([di_data[z]['lbls'].mean() for z in di_data])
mu_cells = pd.concat([di_data[z]['pts'].cell.value_counts().reset_index().rename(columns={'index':'cell','cell':'n'}) for z in di_data])
mu_cells = mu_cells[mu_cells.cell.isin(cells)].n.sum() / len(di_data)
print('Error: %0.2f' % (mu_pixels * n_pixels**2 / fillfac - mu_cells))
b0 = np.log(mu_pixels / (1 - mu_pixels))


num_cell = df_cells[cells].sum(1).astype(int)
num_agg = df_cells.drop(columns=['id']).sum(1).astype(int)
df_cells = pd.DataFrame({'id':df_cells.id,'num_cell':num_cell,'num_agg':num_agg})
df_cells = df_cells.assign(ratio = lambda x: x.num_cell/x.num_agg)
df_cells = df_cells.sort_values('ratio').reset_index(None,True)
df_cells.id = df_cells.id.str.split('\\.',1,True).iloc[:,0]
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
gg_cells = (ggplot(df_cells.melt('id',None,'tt'),aes(x='value')) + 
    theme_bw() + labs(x='Value',y='Frequency') + 
    facet_wrap('~tt',scales='free') + 
    geom_histogram(bins=20,color='black',fill='red',alpha=0.5) + 
    theme(axis_title_x=element_blank(),subplots_adjust={'wspace': 0.15}))
gg_cells.save(os.path.join(dir_figures,'gg_cells_'+'_'.join(cells)+'.png'),width=12,height=4)
np.sum(df_cells==0,0)
print(df_cells)


# Load the model
torch.manual_seed(1234)
mdl = UNet(n_channels=3, n_classes=1, bl=np, batchnorm=True)
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
optimizer = torch.optim.Adagrad(params=mdl.parameters(), lr=lr)

tnow = time()
# Loop through images and make sure they average number of cells equal
mat = np.zeros([len(ids_tissue), 2])
for ii, idt in enumerate(ids_tissue):
    print('ID-tissue %s (%i of %i)' % (idt, ii + 1, len(ids_tissue)))
    tens = img2tensor(device)([di_data[idt]['img'],di_data[idt]['lbls']])[0]
    tens = tens.reshape([1,3,n_pixels,n_pixels]) / 255
    arr = torch2array(tens)
    with torch.no_grad():
        logits = mdl(tens)
        ncl = logits.cpu().mean().numpy()+0
        nc = torch.sigmoid(logits).cpu().sum().numpy()+0
        mat[ii] = [nc, ncl]
print('Script took %i seconds' % (time() - tnow))
print(mat.mean(axis=0)); print(mu_pixels*n_pixels**2); print(mu_cells); print(b0)
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
train_params = {'batch': batch, 'shuffle': True}
val_params = {'batch': batch,'shuffle': False}
eval_params = {'batch': batch,'shuffle': True}

multiclass = False

# Traiing
train_transform = transforms.Compose([randomRotate(tol=1e-4), randomFlip(),
                                      img2tensor(device)])
train_data = CellCounterDataset(di=di_data, ids=idt_train, transform=train_transform,
                                multiclass=multiclass)
train_gen = data.DataLoader(dataset=train_data,**train_params)
# Validation
val_transform = transforms.Compose([img2tensor(device)])
val_data = CellCounterDataset(di=di_data, ids=idt_val, transform=val_transform,
                              multiclass=multiclass)
val_gen = data.DataLoader(dataset=val_data,**val_params)
# Eval (all sample)
eval_data = CellCounterDataset(di=di_data, ids=idt_train + idt_val,
                               transform=val_transform, multiclass=multiclass)
eval_gen = data.DataLoader(dataset=eval_data, **eval_params)


tnow = time()
# cn_loss = pd.Series(np.repeat(['train_train','train_eval','val_eval'],2))+'_'+pd.Series(np.tile(['r2','ce'],3))
# print(cn_loss)
# mat_loss = np.zeros([nepoch, len(cn_loss)])
epoch_loss = []
ee, ii = 0, 1
for ee in range(nepoch):
    print('--------- EPOCH %i of %i ----------' % (ee+1, nepoch))

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
            ii_pred_act[kk, 0] = ii_phat[kk].sum() / fillfac
            ii_pred_act[kk, 1] = di_data[ids_batch[kk]]['lbls'].sum() / fillfac
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
            gaussian = np.stack([di_data[ids]['lbls'].copy() for ids in ids_batch], 3)
            ids_seq = list(ids_batch)
            pred_seq = phat.sum(0).sum(0).sum(0) / fillfac
            act_seq = gaussian.sum(0).sum(0).sum(0) / fillfac
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
