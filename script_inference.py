"""
SCRIPT TO EVALUATE THE MODEL PERFORMANCE ACROSS DIFFERENT EPOCHS
"""

import os, pickle
import numpy as np
import pandas as pd
from funs_support import makeifnot, sigmoid, val_plt
import torch

from funs_unet import UNet, find_bl_UNet

# import matplotlib
# if not matplotlib.get_backend().lower() == 'agg':
#     matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# import seaborn as sns

from plotnine import *

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device('cuda' if use_cuda else 'cpu')

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_output, 'figures')
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
dir_snapshot = os.path.join(dir_checkpoint, 'snapshot')
lst_dir = [dir_output, dir_figures, dir_checkpoint, dir_snapshot]
assert all([os.path.exists(path) for path in lst_dir])
dir_inference = os.path.join(dir_figures, 'inference')
makeifnot(dir_inference)

# Get the dates from the snapshot folder
fns_snapshot = pd.Series(os.listdir(dir_snapshot))
dates_snapshot = pd.to_datetime(fns_snapshot.str.split('\\.|\\_',5,True).iloc[:,2:5].apply(lambda x: '-'.join(x),1))
dates2 = pd.Series(dates_snapshot.sort_values(ascending=False).unique())
dnew, dold = dates2[0].strftime('%Y_%m_%d'), dates2[len(dates2)-1].strftime('%Y_%m_%d')
print('The current date is: %s, the oldest is: %s' % (dnew, dold))
# Make folder in inference with the newest date
dir_save = os.path.join(dir_inference, dnew)
makeifnot(dir_save)

###########################################
## --- (1) LOAD DATA AND LOAD MODEL  --- ##

# cell order in the lbls matrix
valid_cells = ['eosinophil', 'neutrophil', 'plasma', 'enterocyte', 'other', 'lymphocyte']
inflam_cells = ['eosinophil', 'neutrophil', 'plasma', 'lymphocyte']
# Load data
di_img_point = pickle.load(open(os.path.join(dir_output, 'di_img_point.pickle'), 'rb'))
ids_tissue = list(di_img_point.keys())
# Image to star eosin and inflam
idx_eosin = np.where(pd.Series(valid_cells) == 'eosinophil')[0]
idx_inflam = np.where(pd.Series(valid_cells).isin(inflam_cells))[0]
for idt in ids_tissue:
    tmp = di_img_point[idt]['lbls'].copy()
    tmp_eosin = tmp[:, :, idx_eosin].sum(2)
    tmp_inflam = tmp[:, :, idx_inflam].sum(2)
    tmp2 = np.dstack([tmp_eosin, tmp_inflam])
    tmp3 = di_img_point[idt]['pts'].copy()
    tmp3 = tmp3[tmp3.cell.isin(inflam_cells)]
    di_img_point[idt]['lbls'] = tmp2
    assert np.abs( tmp3.shape[0] - (tmp_inflam.sum() / 9) ) < 1
    del tmp, tmp2, tmp3

cells = ['eosin','inflam']
dates = [dnew, dold]
# Initialize two models
fn_eosin_new, fn_inflam_new = tuple([os.path.join(dir_snapshot, 'mdl_'+cell+'_'+dnew+'.pt') for cell in cells])
fn_eosin_old, fn_inflam_old = tuple([os.path.join(dir_snapshot, 'mdl_'+cell+'_'+dold+'.pt') for cell in cells])
mdl_eosin_new, mdl_inflam_new = find_bl_UNet(fn_eosin_new, device), find_bl_UNet(fn_inflam_new, device)
mdl_eosin_old, mdl_inflam_old = find_bl_UNet(fn_eosin_old, device), find_bl_UNet(fn_inflam_old, device)
print([z for z in di_mdls['eosin']['2020_08_12'].down4.parameters()][0][10,10,1,1])
# Load the data sources
fn_dat_new, fn_dat_old = tuple([os.path.join(dir_snapshot, 'dat_star_'+date+'.csv') for date in dates])
dat_star = pd.read_csv(os.path.join(dir_snapshot, fn_dat_new))
di_id = dat_star.groupby(['tt','id']).size().reset_index()
di_id = dict(zip(di_id.id, di_id.tt))
# Get the training/validation IDs
idt_val = [k for k,q in di_id.items() if q == 'Validation']
idt_train = [k for k,q in di_id.items() if q == 'Training']

###########################################################
## --- (2) EXAMINE PREDICTED PROBABILITIES ON IMAGE  --- ##

# Loop over validation to make predicted/actual plots
mdl_eosin_new.eval()

holder = []
for idt in idt_val:
    img, gt = di_img_point[idt]['img'].copy(), di_img_point[idt]['lbls'].copy()
    gt_eosin, gt_inflam = gt[:, :, [0]], gt[:, :, [1]]
    timg = torch.tensor(img.transpose(2, 0, 1).astype(np.float32) / 255).to(device)
    timg = timg.reshape([1] + list(timg.shape))
    with torch.no_grad():
        logits_inflam = mdl_inflam_new(timg).cpu().detach().numpy().sum(0).transpose(1,2,0)
        phat_inflam = sigmoid(logits_inflam)
        logits_eosin = mdl_eosin_new(timg).cpu().detach().numpy().sum(0).transpose(1,2,0)
        phat_eosin = sigmoid(logits_eosin)
    pred_inflam, pred_eosin = phat_inflam.sum()/9, phat_eosin.sum()/9
    act_inflam, act_eosin = int(np.round(gt_inflam.sum()/9,0)), int(np.round(gt_eosin.sum()/9,0))
    print('ID: %s -- pred inflam: %i (%i), eosin: %i (%i)' % (idt, pred_inflam, act_inflam, pred_eosin, act_eosin))
    # Seperate eosin from inflam
    thresh_eosin, thresh_inflam = np.quantile(phat_eosin, 0.99), np.quantile(phat_inflam, 0.99)
    print('Threshold inflam: %0.5f, eosin: %0.5f' % (thresh_inflam, thresh_eosin))
    thresh_eosin, thresh_inflam = 0.01, 0.01
    phat = np.dstack([phat_eosin, phat_inflam])
    gt = np.dstack([gt_eosin, gt_inflam])
    idx_cell_other = gt_inflam - gt_eosin > 0
    idx_cell_eosin = gt_eosin > 0
    idx_cell_nothing = gt_inflam == 0
    idx_eosin = phat_eosin > thresh_eosin
    num_other, num_eosin, num_null = idx_cell_other[idx_eosin].sum(), idx_cell_eosin[idx_eosin].sum(), idx_cell_nothing[idx_eosin].sum()
    tmp = pd.DataFrame({'idt':idt, 'other': num_other, 'eosin': num_eosin,
                  'null': num_null, 'tot': np.sum(idx_eosin),
                  'pred':pred_eosin, 'act': act_eosin}, index=[0])
    holder.append(tmp)
    val_plt(img, phat, gt, lbls=['eosin', 'inflam'], path=dir_save,
             thresh=[thresh_eosin, thresh_inflam], fn=idt+'.png')
# Find correlation between...
df_inf = pd.concat(holder).reset_index(None, True).melt(['idt','pred','act','tot'],None, 'tt', 'n')
df_inf = df_inf.merge(df_inf.groupby('idt').n.sum().reset_index().rename(columns={'n':'den'}),'left','idt')
df_inf = df_inf.assign(ratio = lambda x: x.n / x.den).sort_values(['act','tt'])
tmp = df_inf.assign(tt=lambda x: pd.Categorical(x.tt,['null','other','eosin']))
tmp.act = pd.Categorical(tmp.act.astype(str),tmp.act.unique().astype(str))

gg_inf = (ggplot(tmp, aes(x='act',y='ratio',fill='tt')) + theme_bw() +
          geom_bar(stat='identity') + ggtitle('Distribution of points > threshold') +
          labs(y='Percent', x='# of actual eosinophils') +
          scale_fill_discrete(name='Cell type',labels=['Empty','Other Inflam','Eosin']))
gg_inf.save(os.path.join(dir_save,'inf_fp_ratio.png'))

#############################################
## --- (3) COMPARE TO PREVIOUS MODELS  --- ##

# Save models in a dictionary
di_mdls = {'eosin':{dates[0]:mdl_eosin_new, dates[1]:mdl_eosin_old},
           'inflam':{dates[0]:mdl_inflam_new, dates[1]:mdl_inflam_old}}

for ii, idt in enumerate(idt_val):
    print(ii+1)
    img, gt = di_img_point[idt]['img'].copy(), di_img_point[idt]['lbls'].copy()
    gt_eosin, gt_inflam = gt[:, :, [0]], gt[:, :, [1]]
    timg = torch.tensor(img.transpose(2, 0, 1).astype(np.float32) / 255).to(device)
    timg = timg.reshape([1] + list(timg.shape))
    holder_phat, holder_gt, lbls = [], [], []
    for cell in di_mdls:
        for date in di_mdls[cell]:
            lbls.append(cell + '_' + date)
            if cell == 'eosin':
                holder_gt.append(gt_eosin)
            else:
                holder_gt.append(gt_inflam)
            print('Cell: %s, date: %s' % (cell, date))
            with torch.no_grad():
                logits = di_mdls[cell][date].eval()(timg).cpu().detach().numpy().sum(0).transpose(1,2,0)
                phat = sigmoid(logits)
                holder_phat.append(phat)
    phat, gt = np.dstack(holder_phat), np.dstack(holder_gt)
    assert phat.shape == gt.shape
    thresholds = list(np.repeat(0.01, len(lbls)))
    val_plt(img, phat, gt, lbls=lbls, path=dir_save, thresh=thresholds, fn='comp_' + idt+'.png')

