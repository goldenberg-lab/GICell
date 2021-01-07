"""
DETERMINE THE BEST HYPERPARAMETER CONFIGURATION FROM STEP (2) AFTER TRAINING
"""

import os
import numpy as np
import pandas as pd
from plotnine import *
from funs_support import find_dir_cell
import torch
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device('cuda' if use_cuda else 'cpu')

dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')
dir_checkpoint = os.path.join(dir_output, 'checkpoint')

###############################
# ---- (1) LOAD THE DATA ---- #

# Get the cell folders
cellf = pd.Series(os.listdir(dir_checkpoint))
cellf = cellf[cellf.str.contains('^eosinophil')].to_list()
print('Cell folders are: %s' % cellf)

#cf = cellf[0]
holder = []
for cf in cellf:
    fold1 = os.path.join(dir_checkpoint, cf)
    # Get dates
    dates = pd.Series(os.listdir(fold1))
    dates = dates[dates.str.contains('^[0-9]{4}')].to_list()
    print('There are %i date folders for cell: %s = %s' %
          (len(dates), cf, dates))
    for d in dates:
        fold2 = os.path.join(fold1, d)
        # Get the hyperparameters
        hps = os.listdir(fold2)
        print('There are %i hyperparameter configs' % (len(hps)))
        for h in hps:
            fold3 = os.path.join(fold2, h)
            tmp_hp = pd.read_csv(os.path.join(fold3, 'hyperparameters.csv'))
            tmp_perf = pd.read_csv(os.path.join(fold3, 'mdl_performance.csv'))
            for cn in tmp_hp.columns:
                tmp_perf.insert(tmp_perf.shape[1],cn,tmp_hp[cn][0])
            tmp_perf.insert(0, 'date', d)
            tmp_perf.insert(0, 'cell', cf)
            holder.append(tmp_perf)
# Merge and save
df_perf = pd.concat(holder)
df_perf.to_csv(os.path.join(dir_output, 'df_hp_perf.csv'), index=False)


