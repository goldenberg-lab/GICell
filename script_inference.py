"""
SCRIPT TO EVALUATE THE MODEL PERFORMANCE ACROSS DIFFERENT EPOCHS
"""

import os, pickle
import numpy as np
import pandas as pd
from funs_support import makeifnot, sigmoid, val_plt
import torch

from funs_unet import UNet


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

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_output, 'figures')
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
lst_dir = [dir_output, dir_figures, dir_checkpoint]
assert all([os.path.exists(path) for path in lst_dir])
dir_inference = os.path.join(dir_figures, 'inference')
makeifnot(dir_inference)

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

# Epoch training
# Initialize two models
mdl_eosin, mdl_inflam = UNet(3, 1, 16), UNet(3, 1, 16)
mdl_eosin.to(device)
mdl_inflam.to(device)
mdl_eosin.load_state_dict(torch.load(os.path.join(dir_checkpoint, 'mdl_eosin.pt')))
mdl_inflam.load_state_dict(torch.load(os.path.join(dir_checkpoint, 'mdl_inflam.pt')))

# Load the predicted/actual data
dat_star = pd.read_csv(os.path.join(dir_checkpoint, 'dat_star.csv'))
di_id = dat_star.groupby(['tt','id']).size().reset_index()
di_id = dict(zip(di_id.id, di_id.tt))
# Get the training/validation IDs
idt_val = [k for k,q in di_id.items() if q == 'Validation']
idt_train = [k for k,q in di_id.items() if q == 'Training']

###########################################################
## --- (2) EXAMINE PREDICTED PROBABILITIES ON IMAGE  --- ##

# Loop over validation to make predicted/actual plots
holder = []
for idt in idt_val:
    img, gt = di_img_point[idt]['img'].copy(), di_img_point[idt]['lbls'].copy()
    gt_eosin, gt_inflam = gt[:, :, [0]], gt[:, :, [1]]
    timg = torch.tensor(img.transpose(2, 0, 1).astype(np.float32) / 255).to(device)
    timg = timg.reshape([1] + list(timg.shape))
    with torch.no_grad():
        logits_inflam = mdl_inflam.eval()(timg).cpu().detach().numpy().sum(0).transpose(1,2,0)
        phat_inflam = sigmoid(logits_inflam)
        logits_eosin = mdl_eosin.eval()(timg).cpu().detach().numpy().sum(0).transpose(1,2,0)
        phat_eosin = sigmoid(logits_eosin)
    pred_inflam, pred_eosin = phat_inflam.sum()/9, phat_eosin.sum()/9
    act_inflam, act_eosin = int(np.round(gt_inflam.sum()/9,0)), int(np.round(gt_eosin.sum()/9,0))
    print('ID: %s -- pred inflam: %i, eosin: %i' % (idt, pred_inflam, pred_eosin) )
    # Seperate eosin from inflam
    # np.quantile(phat_eosin, 1 - phat_eosin.sum() / 501 ** 2)
    thresh_eosin, thresh_inflam = np.quantile(phat_eosin, 0.99), np.quantile(phat_inflam, 0.99)
    # gt_inflam[np.where(gt_eosin != 0)] = 0
    # phat_inflam[np.where(phat_eosin > thresh_eosin)] = 0
    phat = np.dstack([phat_eosin, phat_inflam])  #
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

    val_plt(img, phat, gt, lbls=['eosin', 'inflam'], path=dir_inference,
             thresh=[thresh_eosin, thresh_inflam], fn=idt+'.png')
# Find correlation between...
df_inf = pd.concat(holder).reset_index(None, True)
df_inf = df_inf.melt(['idt','pred','act','tot'],None, 'tt', 'n').assign(ratio = lambda x: x.n / x.tot).sort_values(['act','tt'])

xt = range(len(idt_val))
fig, ax = plt.subplots(1, 1, figsize=(7,8))
ax.bar(xt, df_inf[df_inf.tt == 'eosin'].ratio, width=0.5)
ax.bar(xt, df_inf[df_inf.tt == 'other'].ratio, width=0.5)
ax.set_xticks(xt)
ax.set_xticklabels(df_inf.act.unique())
# ax.set_xticklabels(idt_val, rotation=90, size=8)
ax.set_ylabel('Percent')
ax.set_xlabel('# of actual eosinophils')
fig.legend(['eosin', 'other'], loc='right')
fig.suptitle('Distribution of points > threshold', size=16, weight='bold')
fig.subplots_adjust(bottom=0.25)
fig.savefig(os.path.join(dir_figures,'inf_fp_ratio.png'))



