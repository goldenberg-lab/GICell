"""
SCRIPT TO EVALUATE THE MODEL PERFORMANCE ACROSS DIFFERENT EPOCHS
"""

import os
import numpy as np
import pandas as pd
from funs_support import ljoin
import torch

from sklearn.metrics import r2_score

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

def jackknife_r2(act, pred):
    assert len(act) == len(pred)
    n = len(act)
    vec = np.zeros(n)
    r2 = r2_score( act , pred )
    for ii in range(n):
        vec[ii] = r2_score(np.delete(act, ii), np.delete(pred, ii))
    mi, mx = min(vec), max(vec)
    bias = r2 - np.mean(vec)
    mi, mx = mi + bias, mx + bias
    return mi, mx

############################
## --- (1) LOAD DATA  --- ##

di_metric = {'ce':'Cross-Entropy', 'r2':'R-squared'}
di_tt = {'train':'Training', 'val':'Validation'}
# Folders to calculate Eosinophil ratio
di_fold = {'eosinophil_lymphocyte_neutrophil_plasma':'inflam', 'eosinophil':'eosin'}
assert all([os.path.exists(os.path.join(dir_checkpoint,z)) for z in di_fold])
# Find shared epoch
vec_epochs = pd.Series(ljoin([os.listdir(os.path.join(dir_checkpoint,z)) for z in di_fold]))
vec_epochs = vec_epochs[vec_epochs.str.contains('^epoch')].str.split('_',expand=True).iloc[:,1]
vec_epochs = np.sort(vec_epochs[vec_epochs.duplicated()].astype(int).unique())
print('Epochs: %s' % vec_epochs)

# Loop through both folders
holder_perf, holder_cell = [], []
for fold in di_fold:
    print('folder: %s' % fold)
    # Get the training/val cross-entropy
    fold_cell = os.path.join(dir_checkpoint, fold)
    holder = []
    for ee in vec_epochs:
        edir = os.path.join(fold_cell, 'epoch_' + str(ee))
        path = os.path.join(edir, 'df_'+str(ee)+'.csv')
        df_e = pd.read_csv(path).assign(epoch = ee, cell = di_fold[fold])
        holder.append(df_e)
    tmp_cell = pd.concat(holder).reset_index(None, True)
    # Load the train/val performance
    tmp_perf = pd.read_csv(os.path.join(fold_cell, 'mdl_performance.csv')).assign(cell = di_fold[fold])
    holder_cell.append(tmp_cell)
    holder_perf.append(tmp_perf)

# Eosinophil ratio
df_cell = pd.concat(holder_cell).reset_index(None, True)  #[df_cell.epoch == vec_epochs.min()]
dat_act = df_cell.groupby(['id','cell']).act.mean().reset_index().assign(act = lambda x: x.act.astype(int))
dat_act = dat_act.pivot('id','cell','act').reset_index().assign(ratio = lambda x: x.eosin / x.inflam).fillna(0)
di_id = df_cell.groupby(['id', 'tt']).size().reset_index().drop(columns=[0])
di_id = dict(zip(di_id.id, di_id.tt))

tmp1 = df_cell[df_cell.cell == 'inflam'].pivot('id','epoch','pred').reset_index()
tmp2 = df_cell[df_cell.cell == 'eosin'][['id','pred','epoch']]
df_ratio = tmp2.merge(tmp1,'outer','id').melt(tmp2.columns,None,'epoch_inflam','inflam').rename(columns={'pred':'eosin','epoch':'epoch_eosin'})
df_ratio = dat_act[['id','ratio']].merge(df_ratio,'right','id').assign(pred = lambda x: x.eosin / x.inflam)
df_ratio.insert(1,'tt',df_ratio.id.map(di_id))
# Calculate the r-squared by epoch pairs
dat_r2 = df_ratio.groupby(['epoch_eosin','epoch_inflam','tt']).apply(lambda x:
             pd.Series({'r2':r2_score(x.ratio, x.pred),'ci':jackknife_r2(x.ratio.values, x.pred.values)})).reset_index()
dat_r2 = pd.concat([dat_r2.drop(columns=['ci']), pd.DataFrame(np.vstack(dat_r2.ci),columns=['lb','ub'])],1)
dat_r2 = dat_r2.assign(epoch = lambda x: 'e'+x.epoch_eosin.astype(str) + '_' + 'i'+x.epoch_inflam.astype(str))
# dataframe for best combo
dat_epoch = dat_r2.pivot('epoch','tt','r2').reset_index().assign(diff = lambda x: x.Training - x.Validation)
dat_epoch = dat_epoch.sort_values('Validation',ascending=False).reset_index(None,True)
print(dat_epoch.head(1))
print(dat_r2[dat_r2.epoch == 'e100_i50'].T)

# Performance over time
df_perf = pd.concat(holder_perf).reset_index(None, True)
df_perf = df_perf.melt(['epoch','cell'],None,'tmp','val')
df_perf = pd.concat([df_perf.drop(columns=['tmp']),
                     df_perf.tmp.str.split('_',expand=True)],1).rename(columns={0:'metric',1:'tt'})
df_perf = df_perf.assign(metric = lambda x: x.metric.map(di_metric), tt = lambda x: x.tt.map(di_tt))
df_perf = df_perf[df_perf.val >= -0.1]

###############################
## --- (2) MAKE FIGURES  --- ##

# args order: x, y, lower bound, upperbound
def custom_CI(*args, **kwargs):
    data = kwargs.pop('data')
    assert pd.Series(args).isin(data.columns).all()
    x, y, lb, ub = data[args[0]].values, data[args[1]].values, data[args[2]].values, data[args[3]].values
    errors = np.vstack([y - lb, ub - y])
    plt.errorbar(x, y,  yerr=errors, **kwargs)
    plt.scatter(x, y, **kwargs)

# Best Epoch combo
plt.close('all')
g = sns.FacetGrid(dat_r2, hue='tt', height=4.5, aspect=1.2) #[(dat_r2.lb > -1) & (dat_r2.ub <= 1)]
g.map_dataframe(custom_CI, 'epoch', 'r2', 'lb', 'ub')
g.add_legend()
g._legend.set_title('')
yt = np.round(np.arange(-1, 1.01, 0.25),2)
for ax in g.axes.flat:
    ax.set_ylim(-1, 1)
    ax.set_yticks(yt)
    ax.set_yticklabels(yt)
g.set_xticklabels(rotation=90)
g.set_xlabels('Epoch combination')
g.fig.suptitle('R-squared by epoch combination',size=14, weight='bold')
g.fig.subplots_adjust(top=0.9)
g.savefig(os.path.join(dir_figures,'best_epoch.png'))






# Training/validation performance
plt.close('all')
g = sns.FacetGrid(df_perf, row='cell', col='metric', sharex=True, sharey=False, hue='tt')
g.map(plt.scatter, 'epoch', 'val')
g.set_xlabels('Epoch')
g.set_ylabels('Value')
g.savefig(os.path.join(dir_figures,'epoch_perf.png'))

# Ratio prediction by epoch
plt.close('all')
g = sns.FacetGrid(df_cell, col='epoch', sharex=True, sharey=True, hue='tt', col_wrap=3)
g.map(plt.scatter, 'pred', 'act')
g.set_xlabels('Predicted')
g.set_ylabels('Actual')
g.fig.suptitle('Eosinophil ratio performance',size=14,weight='bold')
g.fig.subplots_adjust(top=0.85)
yt = np.round(np.arange(0,0.6,0.1),1)
for ax in g.axes.flat:
    ax.set_ylim(-0.05,0.55)
    ax.set_xlim(-0.05, 0.55)
    ax.set_yticks(yt)
    ax.set_xticks(yt)
    ax.set_yticklabels(yt)
    ax.set_xticklabels(yt)
    ax.plot([0,0.55],[0,0.55],c='black',linestyle='--')
g.savefig(os.path.join(dir_figures,'ratio_perf.png'))









