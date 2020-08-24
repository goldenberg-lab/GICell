"""
SCRIPT TO EVALUATE THE MODEL PERFORMANCE ACROSS DIFFERENT EPOCHS
"""

import os
import shutil
import numpy as np
import pandas as pd
from funs_support import ljoin, makeifnot, jackknife_metric, norm_mse, gg_color_hue
import torch

from sklearn.metrics import r2_score

import matplotlib
if not matplotlib.get_backend().lower() == 'agg':
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
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

idx = pd.IndexSlice

############################
## --- (1) LOAD DATA  --- ##

di_metric = {'ce':'Cross-Entropy', 'r2':'R-squared'}
di_tt = {'train':'Training', 'val':'Validation'}
# Folders to calculate Eosinophil ratio
di_fold = {'eosinophil_lymphocyte_neutrophil_plasma':'inflam', 'eosinophil':'eosin'}
assert all([os.path.exists(os.path.join(dir_checkpoint,z)) for z in di_fold])
di_cell = {'eosin':'Eosinophil', 'inflam':'Inflammatory', 'ratio':'Ratio'}

# Find the shared dates
fold_dates = pd.Series(ljoin([os.listdir(os.path.join(dir_checkpoint,z)) for z in di_fold])).value_counts()
fold_dates = list(fold_dates[(fold_dates == len(di_fold)) & fold_dates.index.str.contains('^[0-9]{4}')].index)
print('Folder dates: %s' % fold_dates)
# Find most recent "date"
dates = pd.to_datetime(fold_dates, format='%Y_%m_%d')
fold_date = fold_dates[dates.argmax()]
date = dates.max()
print('Most recent date: %s (folder: %s)' % (date, fold_date))

# Find shared epoch
vec_epochs = pd.Series(ljoin([os.listdir(os.path.join(dir_checkpoint,z,fold_date)) for z in di_fold]))
vec_epochs = vec_epochs[vec_epochs.str.contains('^epoch')].str.split('_',expand=True).iloc[:,1]
vec_epochs = np.sort(vec_epochs[vec_epochs.duplicated()].astype(int).unique())
print('Epochs: %s' % vec_epochs)

# Loop through both folders
holder_perf, holder_cell = [], []
for fold in di_fold:
    print('folder: %s' % fold)
    # Get the training/val cross-entropy
    fold_cell = os.path.join(dir_checkpoint, fold, fold_date)
    holder = []
    for ee in vec_epochs:
        edir = os.path.join(fold_cell, 'epoch_' + str(ee))
        if os.path.exists(edir):
            path = os.path.join(edir, 'df_'+str(ee)+'.csv')
            df_e = pd.read_csv(path).assign(epoch = ee, cell = di_fold[fold])
            holder.append(df_e)
        else:
            print('Epoch %i does not exist for %s' % (ee, fold))
    tmp_cell = pd.concat(holder).reset_index(None, True)
    holder_cell.append(tmp_cell)

# Eosinophil ratio
df_cell = pd.concat(holder_cell).reset_index(None, True)  #[df_cell.epoch == vec_epochs.min()]
dat_act = df_cell.groupby(['id','cell']).act.mean().reset_index().assign(act = lambda x: x.act.astype(int))
dat_act = dat_act.pivot('id','cell','act').reset_index().assign(ratio = lambda x: x.eosin / x.inflam).fillna(0)
di_id = df_cell.groupby(['id', 'tt']).size().reset_index().drop(columns=[0])
di_id = dict(zip(di_id.id, di_id.tt))
dat_act = dat_act.assign(tt = lambda x: x.id.map(di_id)).sort_values('ratio').reset_index(None,True)

######################################
## --- (2) PERFORMANCE METRICS  --- ##

cn_epoch = ['epoch_eosin','epoch_inflam']

### GET RAW+RAW COUNTS FOR ALL EPOCH COMBINATIONS
# Ratio
tmp1 = df_cell[df_cell.cell == 'inflam'].pivot('id','epoch','pred').reset_index()
tmp2 = df_cell[df_cell.cell == 'eosin'][['id','pred','epoch']]
df_ratio = tmp2.merge(tmp1,'outer','id').melt(tmp2.columns,None,'epoch_inflam','inflam').rename(columns={'pred':'eosin','epoch':'epoch_eosin'}).assign(pred = lambda x: x.eosin / x.inflam).drop(columns=['eosin','inflam'])
df_ratio = df_ratio.merge(dat_act[['id','ratio']],'left','id').rename(columns={'ratio':'act'}).assign(cell='ratio')
df_ratio.insert(1,'tt',df_ratio.id.map(di_id))
# Raw
tmp1 = df_cell[df_cell.cell == 'inflam'].rename(columns={'epoch':'epoch_inflam'}).drop(columns='cell')
tmp2 = df_cell[df_cell.cell == 'eosin'].rename(columns={'epoch':'epoch_eosin'}).drop(columns='cell')
tmp3 = tmp1.merge(tmp2,'outer',['id','tt'],suffixes=('_inflam','_eosin')).melt(['id','tt']+cn_epoch,None,'tmp').assign(lbl=lambda x: x.tmp.str.split('_',1,True).iloc[:,0],
        cell=lambda x: x.tmp.str.split('_',1,True).iloc[:,1]).drop(columns='tmp')
df_rawcell = tmp3.pivot_table('value',['id','tt','cell']+cn_epoch,'lbl').reset_index()
# Merge
df_ratiocell = pd.concat([df_ratio, df_rawcell],0).reset_index(None,True)

### PERFORMANCE FOR CELL/RATIO
dat_metric_ratio = df_ratiocell.groupby(cn_epoch + ['tt','cell']).apply(lambda x:
             pd.Series({'mu_r2':r2_score(x.act, x.pred), 'ci_r2':jackknife_metric(x.act, x.pred,r2_score),'mu_mse':norm_mse(x.act, x.pred),'ci_mse':jackknife_metric(x.act, x.pred,norm_mse)})).reset_index()
dat_metric_ratio = dat_metric_ratio.melt(cn_epoch+['tt','cell'],None,'tmp').assign(moment=lambda x: x.tmp.str.split('_',1,True).iloc[:,0], metric=lambda x: x.tmp.str.split('_',1,True).iloc[:,1]).drop(columns=['tmp'])
dat_metric_ratio = dat_metric_ratio.pivot_table('value',cn_epoch+['tt','cell','metric'],'moment',lambda x: x).reset_index()
dat_metric_ratio = pd.concat([dat_metric_ratio.drop(columns=['ci']), pd.DataFrame(np.vstack(dat_metric_ratio.ci),columns=['lb','ub'])],1)
dat_metric_ratio.mu = dat_metric_ratio.mu.astype(float)
# Create a subset for R2
dat_r2 = dat_metric_ratio[dat_metric_ratio.metric=='r2'].reset_index(None,True).drop(columns='metric')

#############################################
## --- (3) FIND THE BEST EPOCH COMBO   --- ##

cn_cell = ['eosin','inflam','ratio']

# Get mean performance across cells (inflam/eosin/ratio) and set (train/test)
dat_epoch = dat_r2.pivot_table('mu',cn_epoch+['tt'],'cell').reset_index()
dat_epoch = dat_epoch.melt(cn_epoch+['tt']).groupby(cn_epoch).value.apply(lambda x: pd.Series({'mu':x.mean(),'se':x.std()})).reset_index().pivot_table('value',cn_epoch,'level_2').reset_index()
dat_epoch = dat_epoch.assign(ww=lambda x: x.mu+0.25*x.se).sort_values('ww',ascending=False).reset_index(None,True).rename_axis('idx').reset_index()

# Get "best" epoch combo
epoch_eosin, epoch_inflam = tuple([dat_epoch.loc[0,cn] for cn in cn_epoch])
print('Best epoch is eosin: %i, inflam %i' % (epoch_eosin, epoch_inflam))
epoch_star = dat_r2[(dat_r2.epoch_eosin == epoch_eosin) & (dat_r2.epoch_inflam == epoch_inflam)]
print(epoch_star.drop(columns=cn_epoch))

# Make copies of the models for future use
dir_snapshot = os.path.join(dir_checkpoint,'snapshot')
makeifnot(dir_snapshot)
dir_eosin = os.path.join(dir_checkpoint,'eosinophil',fold_date,'epoch_'+str(epoch_eosin))
dir_inflam = os.path.join(dir_checkpoint,'eosinophil_lymphocyte_neutrophil_plasma',fold_date,'epoch_'+str(epoch_inflam))
# Save pred/act by tt
dat_cell_star = pd.concat([df_cell[(df_cell.cell == 'inflam') & (df_cell.epoch == int(epoch_inflam))], df_cell[(df_cell.cell == 'eosin') & (df_cell.epoch == int(epoch_eosin))]],0)
dat_cell_star.to_csv(os.path.join(dir_snapshot, 'dat_star_' + fold_date + '.csv'), index=False)
shutil.copy(os.path.join(dir_eosin, 'mdl_'+str(epoch_eosin)+'.pt'), os.path.join(dir_snapshot, 'mdl_eosin_' + fold_date + '.pt'))
shutil.copy(os.path.join(dir_inflam, 'mdl_'+str(epoch_inflam)+'.pt'), os.path.join(dir_snapshot, 'mdl_inflam_' + fold_date + '.pt'))

###############################
## --- (4) MAKE FIGURES  --- ##

### PREDICATED VS ACTUAL FOR EOSIN/INFLAM/RATIO

df_best = df_ratiocell[(df_ratiocell.epoch_eosin==epoch_eosin) & ((df_ratiocell.epoch_inflam==epoch_inflam))].reset_index(None,True).drop(columns=cn_epoch)

gg_best = (ggplot(df_best, aes(x='pred',y='act',color='tt')) + theme_bw() +
           geom_point() + geom_abline(slope=1,intercept=0,linetype='--') +
           facet_wrap('~cell',scales='free') + labs(x='Predicted',y='Actual') +
           theme(legend_position='bottom',panel_spacing=0.5,legend_box_spacing=0.3) +
           scale_color_discrete(name=' '))
gg_best.save(os.path.join(dir_figures,'gg_scatter_best.png'),height=5,width=10)

### SCATTER BETWEEN PREDICTED EOSIN/INFLAM

tmp1 = df_rawcell[(df_rawcell.epoch_eosin==epoch_eosin) & ((df_rawcell.epoch_inflam==epoch_inflam))].drop(columns=cn_epoch+['act'])
tmp2 = df_ratio[(df_ratio.epoch_eosin==epoch_eosin) & ((df_ratio.epoch_inflam==epoch_inflam))].drop(columns=cn_epoch+['pred','cell']).rename(columns={'act':'ratio'})
tmp = tmp1.pivot_table('pred',['id','tt'],'cell').reset_index().merge(tmp2,'left',['id','tt'])

gg_heat = (ggplot(tmp, aes(x='eosin',y='inflam',fill='ratio',shape='tt')) +
           theme_bw() + geom_point(size=3) +
           scale_fill_cmap(cmap_name='YlGnBu',name='True eosin ratio') +
           ggtitle('Predicted cell counts and actual ratio') +
           labs(x='Predicted Eosinophils',y='Predicted Inflammatory') +
           scale_shape_discrete(name='Set',labels=['Test','Validation']))
gg_heat.save(os.path.join(dir_figures,'gg_scatter_pred.png'),height=7,width=8)

### R-SQUARED GRID BY EPOCH
tmp = dat_r2.assign(mu=lambda x: np.where(x.mu<0,0,x.mu)).merge(dat_epoch[cn_epoch + ['idx']],'left',cn_epoch).assign(idx=lambda x: np.where(x.idx <= 4, (x.idx+1).astype(str), ''))
breaks = pd.Series(np.arange(0,1.01,0.25))
gg_r2_grid = (ggplot(tmp,aes(x='epoch_eosin',y='epoch_inflam',fill='mu')) +
         theme_bw() + geom_tile(aes(width=35,height=35)) +
         facet_grid('cell~tt',labeller=label_both) +
         scale_fill_gradient2(low='cornflowerblue',high='indianred',mid='grey',
                              midpoint=0.5, breaks=breaks,limits=[0,1],name='R2') +
         labs(y='Epoch: Inflammatory',x='Epoch: Eosinophil') +
         ggtitle('R-squared by epoch combination') +
         geom_text(aes(x='epoch_eosin',y='epoch_inflam',label='idx'),size=10))
gg_r2_grid.save(os.path.join(dir_figures,'gg_r2_grid.png'),height=8,width=10)

### R-SQUARED TREND BY EPOCH EOSON
tmp = dat_r2.assign(tt_cell=lambda x: x.tt + '_' + x.cell)#.drop(columns=['tt','cell'])
ulbl = pd.Series(list(tmp.tt_cell.unique()))
ushapes = np.where(ulbl.str.contains('Training'),'o','s')
ucolorz = np.tile(gg_color_hue(dat_r2.cell.unique().shape[0]),2)
ustr = ulbl.str.split('_',1,True).apply(lambda x: di_cell[x[1]] + ', ('+x[0]+')', 1)
idx = np.argsort(ustr)
ulbl, ushapes, ucolorz, ustr = ulbl[idx], ushapes[idx], ucolorz[idx], ustr[idx]
tmp.tt_cell = pd.Categorical(tmp.tt_cell, ulbl).map(dict(zip(ulbl,ustr)))

gg_r2_trend = (ggplot(tmp,aes(x='epoch_inflam',y='mu.clip(0)',color='tt_cell',shape='tt_cell')) +
               theme_bw() + geom_point() + geom_line(aes(linetype='tt')) +
               labs(x='Epoch: Inflammatory',y='R-squared') +
               facet_wrap('~epoch_eosin',ncol=4) + guides(linetype=False) +
               ggtitle('Trend in R-squared by epoch') +
               scale_color_manual(name='Label',values=ucolorz) +
               scale_shape_manual(name='Label',values=ushapes))
gg_r2_trend.save(os.path.join(dir_figures,'gg_r2_trend.png'),height=8,width=10)

### EOSIN RATIO BY ID
tmp = dat_act.copy()
tmp.id = pd.Categorical(tmp.id, tmp.sort_values('ratio').id)
gg_id_ratio = (ggplot(tmp, aes(x='id',y='ratio',color='tt')) + theme_bw() +
                      geom_point() + labs(y='Eosinophilic ratio') +
                      ggtitle('Distribution of eosinophilic ratios') +
                      theme(axis_title_x=element_blank(),panel_grid_major_x=element_blank(),
                            axis_text_x=element_text(angle=90,size=6),legend_position=(0.3,0.7)))
gg_id_ratio.save(os.path.join(dir_figures,'gg_id_ratio.png'),height=5,width=9)

### PREDICTED/ACTUAL BY EPOCH
gg_pred = (ggplot(df_cell, aes(x='pred',y='act', color='tt')) + theme_bw() + geom_point() +
           labs(x='Predicted',y='Actual') + ggtitle('Eosinophil/Inflammatory prediction') +
           geom_abline(intercept=0,slope=1,color='black',linetype='--') +
           theme(legend_position='bottom',panel_spacing=0.5) +
           facet_wrap('~cell+epoch',scales='free',labeller=label_both,ncol=5) +
           scale_color_discrete(name=' '))
gg_pred.save(os.path.join(dir_figures,'gg_scatter_cells.png'),width=16,height=10)

### RATIO BY EPOCH COMBO
tmp = df_ratio[((df_ratio.epoch_eosin // 50) % 2 == 0) & ((df_ratio.epoch_inflam // 50) % 2 == 0)]

gg_ratio = (ggplot(tmp, aes(x='pred',y='act',color='tt')) + theme_bw() + geom_point() +
           labs(x='Predicted',y='Actual') + ggtitle('Eosinophil ratio prediction comparison') +
           theme(panel_spacing=0.10, panel_grid_major=element_blank(),
                 axis_text_x=element_text(angle=90)) +
           facet_grid('epoch_eosin~epoch_inflam',labeller=label_both) +
           geom_abline(intercept=0,slope=1,color='black',linetype='--'))
gg_ratio.save(os.path.join(dir_figures,'gg_scatter_ratio.png'),height=9,width=10)

#################################
## --- (5) MODEL STACKING  --- ##

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

Xdat = df_best.pivot_table(['pred','act'],['id','tt'],'cell').reset_index()
Xdat = pd.concat([Xdat[['id','tt']].droplevel(1,1),
           Xdat.loc[:,pd.IndexSlice['act',['ratio']]].droplevel(0,1),
           Xdat.loc[:,pd.IndexSlice['pred',['eosin','inflam']]].droplevel(0,1)],1)

cn_X = ['eosin','inflam']
Xtrain = Xdat.loc[Xdat.tt=='Training',cn_X].values
ytrain = Xdat.loc[Xdat.tt=='Training','ratio'].values
enc = StandardScaler().fit(Xtrain)
Xtrain, X = enc.transform(Xtrain), enc.transform(Xdat[cn_X].values)
y = Xdat.loc[:,'ratio'].values
gpr.fit(Xtrain, ytrain)
print(gpr.kernel_)
print(r2_score(y,gpr.predict(X)))

res = pd.DataFrame({'y':y,'tt':Xdat.tt,'yhat':gpr.predict(X),'se':gpr.predict(X,True)[1]})

gg_gpr = (ggplot(res,aes(x='yhat',y='y',color='tt')) + theme_bw() +
          geom_point() + geom_abline(slope=1,intercept=0,linetype='--') +
          ggtitle('Gaussian Process model'))
gg_gpr.save(os.path.join(dir_figures,'gg_gpr.png'),height=5,width=5)

