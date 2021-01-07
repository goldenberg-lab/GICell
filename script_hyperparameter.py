"""
DETERMINE THE BEST HYPERPARAMETER CONFIGURATION FROM STEP (2) AFTER TRAINING
"""

import os
import shutil
import pandas as pd
from sklearn.metrics import r2_score
from funs_support import find_dir_cell

dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
dir_snapshot = os.path.join(dir_checkpoint, 'snapshot')

di_cell = {'eosinophil_lymphocyte_neutrophil_plasma':'Inflammatory',
           'eosinophil':'Eosinophil'}

###############################
# ---- (1) LOAD THE DATA ---- #

# Get the cell folders
cellf = pd.Series(os.listdir(dir_checkpoint))
cellf = cellf[cellf.str.contains('^eosinophil')].to_list()
print('Cell folders are: %s' % cellf)

holder = []
holder2 = []
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
            # Get the epochs
            fn_epochs = pd.Series(os.listdir(fold3))
            fn_epochs = fn_epochs[fn_epochs.str.contains('^epoch')].to_list()
            print('There are %i epochs' % (len(fn_epochs)))
            for e in fn_epochs:
                fold4 = os.path.join(fold3, e)
                epoch = e.split('_')[1]
                path4 = os.path.join(fold4, 'df_'+str(epoch)+'.csv')
                tmp4 = pd.read_csv(path4)
                tmp4 = tmp4.query('tt=="Validation"').drop(columns=['ce','tt'])
                tmp_r2 = r2_score(tmp4.act, tmp4.pred)
                tmp_slice = pd.DataFrame({'cell':cf,'fold':fold4,'r2':tmp_r2,'date':d},index=[0])
                holder2.append(tmp_slice)

# Determine the "best" folder
df_fold = pd.concat(holder2)
df_fold = df_fold.sort_values(['cell','r2'],ascending=[True,False]).reset_index(None, True)
df_fold = df_fold.groupby('cell').head(1).reset_index(None, True)
df_fold.cell = df_fold.cell.map(di_cell)
print(df_fold.T)
for ii, rr in df_fold.iterrows():
    fold, date, cell = rr['fold'], rr['date'], rr['cell']
    # Copy the point predictions and model
    fn_fold = pd.Series(os.listdir(fold))
    fn_df = fn_fold[fn_fold.str.contains('^df\\_')].to_list()[0]
    fn_mdl = fn_fold[fn_fold.str.contains('^mdl\\_')].to_list()[0]
    path_df = os.path.join(fold, fn_df)
    path_mdl = os.path.join(fold, fn_mdl)
    dest_df = os.path.join(dir_snapshot, 'df_'+cell+'_'+date+'.csv')
    dest_mdl = os.path.join(dir_snapshot, 'mdl_' + cell + '_' + date + '.csv')
    shutil.copy(path_df, dest_df)
    shutil.copy(path_mdl, dest_mdl)

# Merge and save tensorboard
df_perf = pd.concat(holder)
df_perf.to_csv(os.path.join(dir_output, 'df_hp_perf.csv'), index=False)











