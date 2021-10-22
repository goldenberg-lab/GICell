# Script to rename models from hash to best

import os
import shutil
import pandas as pd
import numpy as np
from funs_support import find_dir_cell, hash_hp


dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_checkpoint = os.path.join(dir_output,'checkpoint')

#########################
# --- (1) FIND DATA --- #

# (i) Load best HP file
df_hp = pd.read_csv(os.path.join(dir_output, 'hp_best.csv'))

# (ii) Find the hashed file equivalent
cn_hp = ['lr', 'p', 'batch', 'nepoch']

di_hp = {}
for ii, rr in df_hp.iterrows():
    cell, lr, p, batch, epoch = rr['cell'], rr['lr'], rr['p'], rr['batch'], rr['epoch']
    df_slice = pd.DataFrame({'lr':lr, 'p':p, 'batch':batch, 'nepoch':100},index=[0])
    code_hash = hash_hp(df_slice, cn_hp)
    di_hp[cell] = code_hash

# (iii) Find the associated model
for cell in os.listdir(dir_checkpoint):
    dir_cell = os.path.join(dir_checkpoint, cell)
    fn_pkl = pd.Series(os.listdir(dir_cell))
    fn_pkl = fn_pkl[fn_pkl.str.contains('^[0-9]{1,64}')]
    fn_hash = fn_pkl.str.replace('.pkl','',regex=False).apply(int)
    pkl_hash = fn_pkl[fn_hash == di_hp[cell]]
    assert len(pkl_hash) == 1, 'Found more than one pickle file!'
    fn_cell = pkl_hash.values[0]
    path_from = os.path.join(dir_cell, fn_cell)
    path_to = os.path.join(dir_cell, 'best_%s.pkl' % cell)
    shutil.copy(src=path_from, dst=path_to)
