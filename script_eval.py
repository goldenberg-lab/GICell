"""
SCRIPT TO EVALUATE THE MODEL PERFORMANCE ACROSS DIFFERENT EPOCHS
"""

import os, re, shutil
import numpy as np
import pandas as pd
from funs_support import ljoin, makeifnot, jackknife_r2
import torch
from datetime import datetime

from sklearn.metrics import r2_score

import matplotlib
if not matplotlib.get_backend().lower() == 'agg':
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
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
lst_dir = [dir_output, dir_figures, dir_checkpoint]
assert all([os.path.exists(path) for path in lst_dir])
dir_inference = os.path.join(dir_figures, 'inference')
makeifnot(dir_inference)


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
    # tmp_perf = pd.read_csv(os.path.join(fold_cell, 'mdl_performance.csv')).assign(cell = di_fold[fold])
    holder_cell.append(tmp_cell)
    # holder_perf.append(tmp_perf)

# Eosinophil ratio
df_cell = pd.concat(holder_cell).reset_index(None, True)  #[df_cell.epoch == vec_epochs.min()]
dat_act = df_cell.groupby(['id','cell']).act.mean().reset_index().assign(act = lambda x: x.act.astype(int))
dat_act = dat_act.pivot('id','cell','act').reset_index().assign(ratio = lambda x: x.eosin / x.inflam).fillna(0)
di_id = df_cell.groupby(['id', 'tt']).size().reset_index().drop(columns=[0])
di_id = dict(zip(di_id.id, di_id.tt))
dat_act = dat_act.assign(tt = lambda x: x.id.map(di_id)).sort_values('ratio').reset_index(None,True)

tmp1 = df_cell[df_cell.cell == 'inflam'].pivot('id','epoch','pred').reset_index()
tmp2 = df_cell[df_cell.cell == 'eosin'][['id','pred','epoch']]
df_ratio = tmp2.merge(tmp1,'outer','id').melt(tmp2.columns,None,'epoch_inflam','inflam').rename(columns={'pred':'eosin','epoch':'epoch_eosin'})
df_ratio = dat_act[['id','ratio']].merge(df_ratio,'right','id').assign(pred = lambda x: x.eosin / x.inflam)
df_ratio.insert(1,'tt',df_ratio.id.map(di_id))
# Calculate the r-squared by epoch pairs
dat_r2 = df_ratio.groupby(['epoch_eosin','epoch_inflam','tt']).apply(lambda x:
             pd.Series({'r2':r2_score(x.ratio, x.pred), 'ci':jackknife_r2(x.ratio.values, x.pred.values)})).reset_index()
dat_r2 = pd.concat([dat_r2.drop(columns=['ci']), pd.DataFrame(np.vstack(dat_r2.ci),columns=['lb','ub'])],1)
dat_r2 = dat_r2.assign(epoch = lambda x: 'e'+x.epoch_eosin.astype(str) + '_' + 'i'+x.epoch_inflam.astype(str))
# dataframe for best combo
dat_epoch = dat_r2.pivot('epoch','tt','r2').reset_index().assign(desc = lambda x: np.abs(x.Training - x.Validation),perf=lambda x: (x.Training + x.Validation)/2)
dat_epoch = dat_epoch.assign(score=lambda x: x.perf - x.desc/2).sort_values('score',ascending=False).reset_index(None,True)
# Get "best" epoch combo
epoch_star = dat_epoch.loc[0,'epoch']
epoch_eosin, epoch_inflam = tuple([int(z[1:]) for z in epoch_star.split('_')])
print('Best epoch is eosin: %i, inflam %i' % (epoch_eosin, epoch_inflam))
lb_star = dat_r2[dat_r2.epoch == epoch_star].lb.min()
ub_star = dat_r2[dat_r2.epoch == epoch_star].ub.max()
r2_star = dat_r2[dat_r2.epoch == epoch_star].r2.mean()
print('The best epoch is: %s (r2=%0.1f%%, (%0.1f%%, %0.1f%%))' %
      (epoch_star, r2_star*100, lb_star*100, ub_star*100))

# Make copies of the models for future use
dir_snapshot = os.path.join(dir_checkpoint,'snapshot')
dir_eosin = os.path.join(dir_checkpoint,'eosinophil','epoch_'+str(epoch_eosin))
dir_inflam = os.path.join(dir_checkpoint,'eosinophil_lymphocyte_neutrophil_plasma','epoch_'+str(epoch_inflam))
# Save pred/act by tt
dat_cell_star = pd.concat([df_cell[(df_cell.cell == 'inflam') & (df_cell.epoch == int(epoch_inflam))], df_cell[(df_cell.cell == 'eosin') & (df_cell.epoch == int(epoch_eosin))]],0)

yymmdd = datetime.now().strftime('%Y_%m_%d')
dat_cell_star.to_csv(os.path.join(dir_snapshot, 'dat_star_' + yymmdd + '.csv'), index=False)
shutil.copy(os.path.join(dir_eosin, 'mdl_'+str(epoch_eosin)+'.pt'), os.path.join(dir_snapshot, 'mdl_eosin_' + yymmdd + '.pt'))
shutil.copy(os.path.join(dir_inflam, 'mdl_'+str(epoch_inflam)+'.pt'), os.path.join(dir_snapshot, 'mdl_inflam_' + yymmdd + '.pt'))

# # Performance over time
# df_perf = pd.concat(holder_perf).reset_index(None, True)
# df_perf = df_perf.melt(['epoch','cell'],None,'tmp','val')
# df_perf = pd.concat([df_perf.drop(columns=['tmp']),
#                      df_perf.tmp.str.split('_',expand=True)],1).rename(columns={0:'metric',1:'tt'})
# df_perf = df_perf.assign(metric = lambda x: x.metric.map(di_metric), tt = lambda x: x.tt.map(di_tt))
# df_perf = df_perf[df_perf.val >= -0.1]

# Is r-squared different for the non-zero values?
print(df_ratio[df_ratio.ratio > 0].groupby(['epoch_eosin','epoch_inflam','tt']).apply(lambda x: r2_score(x.ratio, x.pred)).reset_index().pivot_table(0,['epoch_eosin','epoch_inflam'],'tt').reset_index().sort_values('Training',ascending=False))
# Currently not much of a performance gain for training...

#################################
## --- (2) ANALYZE RATIOS  --- ##

def get_split(x,pat='\\s',k=0,n=5):
    return x.str.split(pat,n,True).iloc[:,k]

# Let's look at predictions/act above 0.75
df_best = df_ratio[(df_ratio.epoch_eosin==epoch_eosin) & ((df_ratio.epoch_inflam==epoch_inflam))].reset_index(None,True).drop(columns=['epoch_eosin','epoch_inflam'])
df_best = df_best.drop(columns='ratio').rename(columns={'pred':'ratio'}).merge(dat_act,'left',['id','tt'],suffixes=('_pred','_act'))
df_best_long = df_best.melt(['id','tt'],None,'tmp').assign(metric=lambda x: get_split(x.tmp,'_',0),gt=lambda x: get_split(x.tmp,'_',1)).drop(columns='tmp')
df_best_wide = df_best_long.pivot_table('value',['id','tt','metric'],'gt').reset_index()

gg_best = (ggplot(df_best_wide, aes(x='pred',y='act',color='tt')) + theme_bw() +
           geom_point() + geom_abline(slope=1,intercept=0,linetype='--') +
           facet_wrap('~metric',scales='free') +
           labs(x='Predicted',y='Actual') +
           theme(legend_position='bottom',panel_spacing=0.5,
                 legend_box_spacing=0.3) +
           scale_color_discrete(name=' '))
gg_best.save(os.path.join(dir_figures,'gg_best.png'),height=5,width=10)

idx = pd.IndexSlice
tmp = df_best_long.pivot_table('value',['id','tt'],['metric','gt'])
tmp = pd.concat([tmp.loc[:,idx[['eosin','inflam'],'pred']].droplevel(level=1,axis=1).reset_index(),tmp.loc[:,idx[['ratio'],'act']].droplevel(level=1,axis=1).reset_index().ratio],1)

gg_heat = (ggplot(tmp, aes(x='eosin',y='inflam',fill='ratio',shape='tt')) +
           theme_bw() + geom_point(size=3) +
           scale_fill_cmap(cmap_name='YlGnBu') +
           ggtitle('Predicted cell counts and actual ratio') +
           labs(x='Predicted Eosinophils',y='Actual eosinophils'))
gg_heat.save(os.path.join(dir_figures,'gg_heat.png'),height=7,width=7)

# Can a GP help?
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, RationalQuadratic, PairwiseKernel
from sklearn.preprocessing import StandardScaler

def logit(x,eps=1e-4):
    assert np.all((x >= 0) | (x<= 1))
    x = np.where(x==0, eps, np.where(x==1, 1-eps, x))
    return np.log(x/(1-x))

kern = WhiteKernel(0.1) + 0.1**2 * RBF(1, (0.1, 100))
gpr = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=5,
                               normalize_y=True,random_state=1234)
Xtrain = tmp.loc[tmp.tt=='Training',['eosin','inflam']].values
ytrain = tmp.loc[tmp.tt=='Training','ratio'].values
enc = StandardScaler().fit(Xtrain)
Xtrain = enc.transform(Xtrain)
X = enc.transform(tmp.loc[:,['eosin','inflam']].values)
y = tmp.loc[:,'ratio'].values
gpr.fit(Xtrain, ytrain)
print(gpr.kernel_)
print(r2_score(y,gpr.predict(X)))

res = pd.DataFrame({'y':y,'tt':tmp.tt,'yhat':gpr.predict(X),'se':gpr.predict(X,True)[1]})

gg_gpr = (ggplot(res,aes(x='yhat',y='y',color='tt')) + theme_bw() +
          geom_point() + geom_abline(slope=1,intercept=0,linetype='--') +
          ggtitle('Gaussian Process model'))
gg_gpr.save(os.path.join(dir_figures,'gg_gpr.png'),height=5,width=5)






###############################
## --- (3) MAKE FIGURES  --- ##

# --- FIGURE 1: EOSIN RATIO BY ID --- #
tmp = dat_act.copy()
tmp.id = pd.Categorical(tmp.id, tmp.sort_values('ratio').id)

gg_id = (ggplot(tmp, aes(x='id',y='ratio',color='tt')) + theme_bw() +
         geom_point() + labs(y='Eosinophilic ratio') +
         ggtitle('Distribution of eosinophilic ratios') +
         theme(axis_title_x=element_blank(),panel_grid_major_x=element_blank(),
               axis_text_x=element_text(angle=90,size=6)))
gg_id.save(os.path.join(dir_figures,'id_ratio.png'),height=5,width=8)

# --- FIGURE 2: R2 FOR RATIO BY EPOCH --- #

# args order: x, y, lower bound, upperbound
def custom_CI(*args, **kwargs):
    data = kwargs.pop('data')
    assert pd.Series(args).isin(data.columns).all()
    x, y, lb, ub = data[args[0]].values, data[args[1]].values, data[args[2]].values, data[args[3]].values
    errors = np.vstack([y - lb, ub - y])
    plt.errorbar(x, y,  yerr=errors, **kwargs)
    plt.scatter(x, y, **kwargs)

tmp = dat_r2.assign(sig = lambda x: x.epoch == epoch_star)
gg_epoch = (ggplot(tmp,aes(x='epoch',y='r2',color='tt',alpha='sig')) + theme_bw() +
            geom_point() + labs(x='Epoch combination',y='R-squared') +
            geom_linerange(aes(ymin='lb',ymax='ub')) +
            scale_alpha_manual(values=[0.3,1]) +
            guides(alpha=False) +
            ggtitle('R-squared by epoch combination') +
            theme(axis_text_x=element_text(angle=90)) +
            scale_y_continuous(limits=[-2,1]))
gg_epoch.save(os.path.join(dir_figures,'best_epoch.png'),width=6, height=6)

# --- FIGURE 3: PREDICTED/ACTUAL BY EPOCH --- #

gg_pred = (ggplot(df_cell, aes(x='pred',y='act', color='tt')) + theme_bw() + geom_point() +
           labs(x='Predicted',y='Actual') + ggtitle('Eosinophil/Inflammatory prediction') +
           geom_abline(intercept=0,slope=1,color='black',linetype='--') +
           theme(legend_position='bottom',panel_spacing=0.5) +
           facet_wrap('~cell+epoch',scales='free',labeller=label_both,ncol=5) +
           scale_color_discrete(name=' '))
gg_pred.save(os.path.join(dir_figures,'pred_act_cells.png'),width=12,height=7)

# --- FIGURE 4: RATIO BY EPOCH COMBO --- #

tmp = df_ratio.assign(epoch = lambda x: 'e'+x.epoch_eosin.astype(str) + '_' + 'i'+x.epoch_inflam.astype(str))

gg_ratio = (ggplot(tmp, aes(x='pred',y='ratio',color='tt')) + theme_bw() + geom_point() +
           labs(x='Predicted',y='Actual') + ggtitle('Eosinophil ratio prediction comparison') +
           theme(panel_spacing=0.10, panel_grid_major=element_blank(),
                 axis_text_x=element_text(angle=90)) +
           facet_grid('epoch_eosin~epoch_inflam',labeller=label_both) +
           geom_abline(intercept=0,slope=1,color='black',linetype='--'))
gg_ratio.save(os.path.join(dir_figures,'pred_act_ratio.png'),height=9,width=10)




