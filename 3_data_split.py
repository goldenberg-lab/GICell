# SCRIPT TO SPLIT TRAINING DATA INTO VALIDATION PORTION

import argparse
from plotnine.themes.themeable import legend_direction, legend_position

from scipy.ndimage.morphology import distance_transform_cdt
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

cn_ord = ['idt','tissue','num','v2']
cn_melt = ['ds','idt_tissue']+cn_ord

df_cells = pd.read_csv(os.path.join(dir_output,'df_cells.csv'))
# Split idt_tissue into idt and tissue
tmp = df_cells['idt_tissue'].str.split('\\_|\\-',3)
tmp_tissue = tmp.apply(lambda x: str_subset(x,'^[A-Z][a-z]'), 1)[0].fillna('Rectum')
tmp = df_cells['idt_tissue'].str.replace('|'.join(tmp_tissue.unique()),'',regex=True)
tmp = tmp.str.replace('\\_{2}','_',regex=True)
tmp = tmp.str.replace('\\_[0-9]{4}\\_[0-9]{2}\\_[0-9]{2}','',regex=True)
tmp_idt = tmp.str.split('\\_',1,True)[0]
df_cells.insert(1,'tissue',tmp_tissue)
df_cells.insert(1,'idt',tmp_idt)
# Calculate inflam + eosin
df_cells = df_cells.assign(inflam=df_cells[inflam_cells].sum(1))
df_cells.rename(columns={'eosinophil':'eosin'}, inplace=True)
df_cells.drop(columns=valid_cells, inplace=True, errors='ignore')
# Assign whether the dataset is a test set
df_cells = df_cells.assign(tt=lambda x: np.where(x.ds.isin(ds_test),'test','train'))
assert not df_cells.groupby('ds').idt_tissue.apply(lambda x: x.duplicated().any()).any(), "Error! idt_tissue is not unique by ds"


#######################################
## --- (2) ASSIGN TRAIN/VAL/TEST --- ##

pct_test = np.mean(df_cells['tt'] == 'test')

print('Proportions: training (%.1f%%), validation (%.1f%%), test (%.1f%%)' % ((1-pct_test)*(1-pval)*100, (1-pct_test)*pval*100, pct_test*100))

# Stratify by eisonophil count
idt_val = df_cells.groupby('ds').apply(lambda x: x.idt_tissue.iloc[stratify_continuous(x.idt_tissue,x.eosin,pval,seed=1)['test']])
idt_val = idt_val.reset_index().drop(columns='level_1').assign(tt2='val')
idt_val = idt_val[~idt_val['ds'].isin(ds_test)]
# Merge and assign
df_cells = df_cells.merge(idt_val,'left')
df_cells = df_cells.assign(tt=lambda x: np.where(x.tt2.isnull(),x.tt,x.tt2))
df_cells.drop(columns='tt2', inplace=True)
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
cn_split = ['ds','idt_tissue','tt']
path_split = os.path.join(dir_output, 'train_val_test.csv')
df_cells[cn_split].to_csv(path_split,index=False)


###############################
## --- (3) AVERAGE CELLS --- ##

cn_gg = ['ds','tt','cell']
long_cells = df_cells.melt(['idt_tissue','tissue']+cn_gg[:-1],['eosin','inflam'],'cell','n')
long_cells['cell'] = long_cells['cell'].str.title()
long_cells['n'] = long_cells['n'].astype(int)
long_cells['tt'] = long_cells['tt'].map(di_tt)
long_cells['ds'] = long_cells['ds'].map(di_ds)
# Aggregate
cell_dist = long_cells.groupby(cn_gg).n.describe()
cell_dist = cell_dist.rename(columns={'25%':'lb','75%':'ub','50%':'mu','count':'n'})
cell_dist = cell_dist.reset_index().drop(columns=['min','max','mean','std'])

posd = pn.position_dodge(0.5)
gg_cell_dist_ds = (pn.ggplot(cell_dist,pn.aes(x='ds',y='mu',color='tt')) + 
    pn.theme_bw() + pn.geom_point(position=posd) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
    pn.labs(y='Median # of cells per image') + 
    pn.ggtitle('Linerange shows IQR') + 
    pn.facet_wrap('~cell',scales='free_y',nrow=2) + 
    pn.scale_color_discrete(name='Dataset') + 
    pn.theme(subplots_adjust={'wspace': 0.15},axis_title_x=pn.element_blank()) )
gg_save('gg_cell_dist_ds.png', dir_figures, gg_cell_dist_ds, 6, 5)


###############################
## --- (4) CELL-SPECIFIC --- ##

long_cells = long_cells.sort_values(cn_gg+['n']).reset_index(None,drop=True)
long_cells['xidx'] = long_cells.groupby(cn_gg).cumcount()

gg_cell_dist_idt = (pn.ggplot(long_cells,pn.aes(x='xidx',y='n',fill='tissue')) + 
    pn.labs(y='# of cells',x='Patient/Tissue') + 
    pn.geom_col(color=None) + pn.theme_bw() + 
    pn.facet_grid('cell~ds',scales='free') + 
    pn.theme(legend_position = (0.5,-0.01),legend_direction='horizontal') + 
    pn.scale_fill_discrete(name='Tissue'))
gg_save('gg_cell_dist_idt.png', dir_figures, gg_cell_dist_idt, 12, 6)
