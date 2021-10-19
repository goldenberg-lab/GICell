# Script to find best hyperparameters in terms of AUROC, CE, AUPRC

import os
import pickle
import pandas as pd
import numpy as np
import plotnine as p9
from funs_support import find_dir_cell
from funs_plotting import gg_save
import gc

# Set up folders
dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
dir_figures = os.path.join(dir_output, 'figures')

def hp_2_str(df, cn):
    assert df.columns.isin(cn).sum() == len(cn)
    holder = pd.Series(np.repeat('',len(df)))
    for i, c in enumerate(cn):
        if i + 1 == len(cn):
            holder += c + '=' + df[c].astype(str)
        else:
            holder += c + '=' + df[c].astype(str) + '_'
    return holder

def get_mi_mx(x):
    return pd.Series({'mi':x.min(), 'mx':x.max()})

##############################
## --- (1) LOAD RESULTS --- ##

cn_hp = ['lr', 'p', 'batch']

holder_ce, holder_pr = [], []
for tt in ['eosin','inflam']:
    print('Model type: %s' % tt)
    dir_tt = os.path.join(dir_checkpoint, tt)
    fn_tt = os.listdir(dir_tt)
    for fn in fn_tt:
        path_fn = os.path.join(dir_tt, fn)
        print(fn)
        with open(path_fn, 'rb') as handle:
            di = pickle.load(handle)
        # Extract the hyperparamets
        df_hp = di['hp']
        if len(df_hp) == 1:
            lr = df_hp['lr'][0]
            p = df_hp['p'][0]
            batch = df_hp['batch'][0]
            # assign and save
            holder_ce.append(di['ce_auc'].assign(cell=tt, lr=lr, p=p, batch=batch))
            holder_pr.append(di['pr'].assign(cell=tt, lr=lr, p=p, batch=batch))
        del di
gc.collect()
# Merge
dat_ce = pd.concat(holder_ce).melt(['cell','epoch']+cn_hp,None,'metric')
dat_pr = pd.concat(holder_pr).melt(['cell','epoch','thresh']+cn_hp,None,'metric')
# Save for analysis later
dat_ce.to_csv(os.path.join(dir_output, 'dat_hp_ce.csv'),index=False)
dat_pr.to_csv(os.path.join(dir_output, 'dat_hp_pr.csv'),index=False)

# # Assign total hp
# dat_ce.insert(0,'hp', hp_2_str(dat_ce, cn_hp))
# dat_pr.insert(0,'hp', hp_2_str(dat_pr, cn_hp))

###############################
## --- (2) FIND BEST AUC --- ##

best_auc = dat_ce.query('metric=="auc"').reset_index(None,True)
print(best_auc.loc[best_auc.groupby('cell').value.idxmax()])


##############################
## --- (3) PLOT RESULTS --- ##

epoch_min = 10
cn_gg = ['cell']+cn_hp

di_moment = {'ce':{'mi':'best', 'mx':'worst'}, 'auc':{'mi':'worst', 'mx':'best'}}

# --- (i) Cross entropy + AUC --- #
for metric in ['ce','auc']:
    tmp_df = dat_ce.query('metric==@metric & epoch>=@epoch_min')
    tmp_df = tmp_df.assign(epoch=lambda x: x.epoch-epoch_min+1)
    # Index to worst
    tmp_moment = tmp_df.query('epoch==1').groupby('cell').value.apply(get_mi_mx).reset_index()
    tmp_moment = tmp_moment.rename(columns={'level_1':'moment'}).pivot('cell','moment','value').reset_index()
    tmp_moment.rename(columns=di_moment[metric],inplace=True)
    tmp_df = tmp_df.merge(tmp_moment,'left').assign(idx=lambda x: x.value/x.worst*100)
    tmp_df.drop(columns=['value','worst','best', 'metric'], inplace=True)
    # Horizontal lines at best index value
    tmp_vlines = tmp_df.groupby('cell').idx.apply(get_mi_mx).reset_index()
    tmp_vlines = tmp_vlines.rename(columns={'level_1':'moment'}).pivot('cell','moment','idx').reset_index()
    tmp_vlines.rename(columns=di_moment[metric],inplace=True)
    # Create titles and files names
    tmp_fn = 'gg_' + metric + '_val.png'
    tmp_gtit = 'metric = %s\nDashed lines show "best"\nBurn-in epochs=%i' % (metric,epoch_min)
    tmp_gtit += '\n best: ' + tmp_moment.assign(lbl=lambda x: x.cell+'='+x.best.round(2).astype(str)).lbl.str.cat(sep=', ')
    # Plot and save
    tmp_gg = (p9.ggplot(tmp_df, p9.aes(x='epoch',y='idx',color='cell')) + 
        p9.theme_bw() + p9.geom_line() + 
        p9.ggtitle(tmp_gtit) + 
        p9.geom_hline(p9.aes(yintercept='best',color='cell'),linetype='--',
                        data=tmp_vlines,inherit_aes=False) + 
        p9.facet_grid('lr+p~batch',labeller=p9.label_both) + 
        p9.labs(x='Epoch',y='Value (100==epoch 1)'))
    gg_save(tmp_fn, dir_figures, tmp_gg, 10, 10)





