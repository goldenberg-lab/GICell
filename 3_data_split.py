# SCRIPT TO SPLIT TRAINING DATA INTO VALIDATION PORTION

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pval', type=float, default=0.2, help='Percent of non-test folders to apply random split to')
parser.add_argument('--ds_test', nargs='+', help='Folders that should be reserved for testing')
args = parser.parse_args()
pval, ds_test = args.pval, args.ds_test
print('args: %s' % args)

# # For debugging
# pval, ds_test = 0.2, 'oscar dua 70608'.split(' ')

# Load modules
import os
import numpy as np
import pandas as pd
import plotnine as pn
from scipy import stats
from cells import inflam_cells, valid_cells, di_ds, di_tt
from funs_plotting import gg_save
from funs_stats import stratify_continuous
from funs_support import find_dir_cell, makeifnot, str_subset

# Set up folders
dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')
dir_labels = os.path.join(dir_figures, 'labels')
[makeifnot(path) for path in [dir_figures, dir_labels]]

seednum = 1234
idx = pd.IndexSlice


###########################
## --- (1) LOAD DATA --- ##

df_cells = pd.read_csv(os.path.join(dir_output,'df_cells.csv'))
# Split idt into id and tissue
tmp_tissue = df_cells['idt'].str.split('\\_|\\-',3)
tmp_tissue = tmp_tissue.apply(lambda x: str_subset(x,'^[A-Z][a-z]'), 1)[0].fillna('Rectum')
tmp_tissue = tmp_tissue.str.replace('[^A-Za-z]','',regex=True)
u_tissue = tmp_tissue.unique()
print('Unique tissues: %s' % u_tissue)
tmp_id = df_cells['idt'].str.replace('|'.join(u_tissue),'',regex=True)
tmp_id = tmp_id.str.replace('\\_{2}','_',regex=True)
tmp_id = tmp_id.str.replace('\\_[0-9]{4}\\_[0-9]{2}\\_[0-9]{2}','',regex=True)
df_cells.insert(1,'tissue',tmp_tissue)
df_cells.insert(1,'id',tmp_id)

# Calculate inflam + eosin
df_cells = df_cells.assign(inflam=df_cells[inflam_cells].sum(1))
df_cells.rename(columns={'eosinophil':'eosin'}, inplace=True)
df_cells.drop(columns=valid_cells, inplace=True, errors='ignore')
# Assign whether the dataset is a test set
df_cells = df_cells.assign(tt=lambda x: np.where(x.ds.isin(ds_test),'test','train'))
check1 = df_cells.groupby(['ds','idt']).size().max() == 1
check2 = df_cells.groupby(['ds','id','tissue']).size().max() == 1
assert check1 & check2, "Error! idt is not unique by ds"

# Count number of unique patients
u_idt = pd.Series(df_cells['idt'].unique())
u_patients = u_idt.str.replace('cleaned_','').str.split('_',1,True)[0]
df_patients = pd.DataFrame({'idt':u_idt, 'patient':u_patients})
df_patients = df_cells[['ds','idt','tissue']].merge(df_patients)
# Find treatment naive patients
u_patient_treated = df_patients.query('tissue!="Rectum"')['patient'].unique()
u_patient_untreated = np.setdiff1d(u_patients, u_patient_treated)
print('A total of %i unique patients (%i untreated, %i treated)' % (u_patients.shape[0], u_patient_untreated.shape[0], u_patient_treated.shape[0]))


#######################################
## --- (2) ASSIGN TRAIN/VAL/TEST --- ##

pct_test = np.mean(df_cells['tt'] == 'test')

print('Proportions: training (%.1f%%), validation (%.1f%%), test (%.1f%%)' % ((1-pct_test)*(1-pval)*100, (1-pct_test)*pval*100, pct_test*100))

# Stratify by eisonophil count
fn_val = df_cells.groupby('ds').apply(lambda x: x['idt'].iloc[stratify_continuous(x['idt'],x['eosin'],pval,seed=1)['test']])
fn_val = fn_val.reset_index().drop(columns='level_1').assign(tt2='val')
fn_val = fn_val[~fn_val['ds'].isin(ds_test)]
# Merge and assign
df_cells = df_cells.merge(fn_val,'left')
df_cells = df_cells.assign(tt=lambda x: np.where(x.tt2.isnull(),x.tt,x.tt2))
df_cells.drop(columns='tt2', inplace=True)
print(df_cells.groupby(['tt','ds']).size())
# Check
ttest_check = df_cells.query('tt != "test"').groupby(['ds','tt']).eosin.apply(lambda x: pd.DataFrame({'mu':x.mean(),'se':x.std(ddof=1),'n':len(x)},index=[0]))
ttest_check = ttest_check.reset_index().drop(columns='level_2')
ttest_check = ttest_check.pivot_table(['mu','se','n'],'ds','tt')
mus = ttest_check.loc[:,idx['mu']]
vars = ttest_check.loc[:,idx['se']]**2
ns = ttest_check.loc[:,idx['n']]
dmus = mus.assign(diff=lambda x: x.train - x.val)['diff']
dens = np.sqrt((vars*(ns-1)).sum(1) / (ns.sum(1)-2) * (1/ns).sum(1))
tscores = dmus / dens
t_pval = 2*(1-stats.t(df=ns.sum(1)-1).cdf(tscores.abs()))
assert t_pval.min() > 0.05, "Warning, t-test was rejected!"
# Save for later
cn_split = ['ds','idt','id','tissue','tt']
path_split = os.path.join(dir_output, 'train_val_test.csv')
df_sets = df_cells[cn_split].copy()
assert not df_sets.groupby('ds').apply(lambda x: x['idt'].duplicated().any()).any(), 'duplicated idt'
df_sets.to_csv(path_split,index=False)


###############################
## --- (3) AVERAGE CELLS --- ##

cn_gg = ['ds','tt','cell']
long_cells = df_cells.melt(['idt','tissue','h','w']+cn_gg[:-1],['eosin','inflam'],'cell','n')
long_cells['cell'] = long_cells['cell'].str.title()
long_cells['n'] = long_cells['n'].astype(int)
long_cells['tt'] = long_cells['tt'].map(di_tt)
long_cells['ds'] = long_cells['ds'].map(di_ds)
# Cells per pixel
long_cells = long_cells.assign(c2p=lambda x: x.n / (x.h*x.w))

# Aggregate
cell_dist = long_cells.groupby(cn_gg)['c2p'].describe()
cell_dist = cell_dist.rename(columns={'25%':'lb','75%':'ub','50%':'mu','count':'n'})
cell_dist = cell_dist.reset_index().drop(columns=['min','max','mean','std'])
cell_dist.to_csv(os.path.join(dir_figures, 'gg_cell_dist_ds.csv'),index=False)

posd = pn.position_dodge(0.5)
gg_cell_dist_ds = (pn.ggplot(cell_dist,pn.aes(x='ds',y='mu',color='tt')) + 
    pn.theme_bw() + pn.geom_point(position=posd) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
    pn.labs(y='Cells per pixel') + 
    pn.ggtitle('Linerange shows IQR') + 
    pn.facet_wrap('~cell',scales='free_y',nrow=2) + 
    pn.scale_color_discrete(name='Dataset') + 
    pn.theme(subplots_adjust={'wspace': 0.15},axis_title_x=pn.element_blank()) )
gg_save('gg_cell_dist_ds.png', dir_figures, gg_cell_dist_ds, 6, 5)


###############################
## --- (4) CELL-SPECIFIC --- ##

long_cells = long_cells.sort_values(cn_gg+['n']).reset_index(None,drop=True)
long_cells['xidx'] = long_cells.groupby(cn_gg).cumcount()
tmp_vlines = long_cells.groupby(['cell','ds','tt']).xidx.max().reset_index()
tmp_vlines = tmp_vlines.query('tt == "Val"')

gg_cell_dist_idt = (pn.ggplot(long_cells,pn.aes(x='xidx',y='n',fill='tissue')) + 
    pn.labs(y='# of cells',x='Patient/Tissue') + 
    pn.geom_col(color=None) + pn.theme_bw() + 
    pn.ggtitle('Vertical line shows Validation/Training split') + 
    pn.facet_grid('cell~ds',scales='free') + 
    pn.geom_vline(pn.aes(xintercept='xidx'),data=tmp_vlines) + 
    pn.theme(legend_position = (0.5,-0.01),legend_direction='horizontal') + 
    pn.scale_fill_discrete(name='Tissue'))
gg_save('gg_cell_dist_idt.png', dir_figures, gg_cell_dist_idt, 12, 6)
