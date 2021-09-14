"""
LOAD THE MERGED SLICES FROM sript_fullimg.py
"""

import os
import pandas as pd
from funs_support import find_dir_cell, makeifnot
import shutil

dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
dir_snapshot = os.path.join(dir_checkpoint, 'snapshot')
dir_inference = os.path.join(dir_figures, 'inference')
# Get the dates from the snapshot folder
fns_snapshot = pd.Series(os.listdir(dir_snapshot))
fns_snapshot = fns_snapshot[fns_snapshot.str.contains('csv$|pt$')]
dates_snapshot = pd.to_datetime(fns_snapshot.str.split('\\.|\\_', 5, True).iloc[:, 2:5].apply(lambda x: '-'.join(x), 1))
dates2 = pd.Series(dates_snapshot.sort_values(ascending=False).unique())
dnew = dates2[0].strftime('%Y_%m_%d')
print('The current date is: %s' % dnew)
# Make folder in inference with the newest date
dir_save = os.path.join(dir_inference, dnew)

lst_dir = [dir_output, dir_figures, dir_checkpoint, dir_snapshot, dir_inference, dir_save]
assert all([os.path.exists(path) for path in lst_dir])

############################
# -- (1) LOAD AND MERGE -- #

fn_save = pd.Series(os.listdir(dir_save))
fn_save = fn_save[fn_save.str.contains('\\.csv$')].to_list()
dir_slice = os.path.join(dir_save, 'slice')
makeifnot(dir_slice)

holder = []
for fn in fn_save:
    path_fn = os.path.join(dir_save,fn)
    holder.append(pd.read_csv(path_fn))
    shutil.move(path_fn, os.path.join(dir_slice,fn))

df_fullimg = pd.concat(holder).reset_index(None, True).drop(columns = 'fn')
df_fullimg = df_fullimg.assign(ratio = lambda x: x.eosin / x.inflam )
df_fullimg.to_csv(os.path.join(dir_output, 'df_fullimg.csv'), index=False)
