"""
USE THE OUTPUT FROM script_hyperparameter.py TO VISUALIZE AND SELECT THE "BEST" configuration
"""

import os
import numpy as np
import pandas as pd
from funs_support import find_dir_cell
from plotnine import *

# Set up folders
dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')

di_metric = {'r2':'R-squared', 'ce':'Cross-Entropy'}
# (1) Load the in hyperparameter data
df_hp = pd.read_csv(os.path.join(dir_output, 'df_hp_perf.csv'))
df_hp.metric = df_hp.metric.map(di_metric)

# (2) Get the validation results
df_val = df_hp.query('tt=="Validation"').reset_index(None, True).drop(columns=['tt', 'batch'])
# # Remove outliers for R-squared and cross entropy
# df_val = df_val.assign(val = lambda x: np.where(x.metric=='R-squared', x.val.clip(0), x.val.clip(0, 0.03)))

# Remove any columns with no variation
nuc = df_val.apply(lambda x: x.unique().shape[0], 0)
df_val.drop(columns = nuc[nuc == 1].index.to_list(), inplace=True)

# (3) Visualize for R2/cross-entropy and cell type
cell_types = list(df_val.cell.unique())
metric_types = list(di_metric.values())

# Columns to group over
cn_gg = list(df_val.columns.drop(['val','epoch']))
df_val = df_val.sort_values(cn_gg + ['epoch']).reset_index(None, True)

# Smooth values
df_val['smooth'] = df_val.groupby(cn_gg).val.rolling(window=5,center=False).mean().values
# Remove outliers for R-squared and cross entropy
df_val = df_val.assign(smooth = lambda x: np.where(x.metric=='R-squared', x.smooth.clip(0), x.smooth.clip(0, 0.03)))

for cell in cell_types:
    for metric in metric_types:
        print('Cell: %s, metric: %s' % (cell, metric))
        tmp_df = df_val.query('cell==@cell & metric==@metric').reset_index(None, True)
        tmp_fn = 'hp_'+cell+'_'+metric+'.png'
        gg_tit = 'Cell=%s, metric=%s' % (cell, metric)
        gg_tmp = (ggplot(tmp_df, aes(x='epoch', y='smooth', color='num_params')) +
                  theme_bw() + geom_point() + geom_line() +
                  facet_grid('lr~batch_size', labeller=label_both) +
                  ggtitle(gg_tit))
        gg_tmp.save(os.path.join(dir_figures, tmp_fn), width=10, height=8)



