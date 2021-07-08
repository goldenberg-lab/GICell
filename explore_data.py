import os
import numpy as np
import pandas as pd
import plotnine as pn
from plotnine.facets.facet_wrap import facet_wrap
from plotnine.scales.scale_color import scale_color_distiller
from funs_support import find_dir_cell, makeifnot
from funs_plotting import gg_save
from sklearn.model_selection import train_test_split

# Set up folders
dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')
dir_labels = os.path.join(dir_figures, 'labels')
[makeifnot(path) for path in [dir_figures, dir_labels]]

seednum = 1234

di_tt = {'train':'Train','val':'Val','test':'Test','oos':'Cinci'}
di_ds = {'hsk':'SickKids', 'cinci':'Cincinatti'}

###########################
## --- (1) LOAD DATA --- ##

cn_ord = ['idt','tissue','num','v2']
cn_melt = ['ds','idt_tissue']+cn_ord

df_cells = pd.read_csv(os.path.join(dir_output,'df_cells.csv'))
df_cells.idt_tissue[df_cells.idt_tissue.str.contains('v2')]
tmp1 = df_cells.idt_tissue.str.split('\\_|\\-',3,True)
tmp1 = tmp1.rename(columns={0:'idt',3:'v2'}).assign(v2=lambda x: np.where(x.v2.isnull(),False,True))
tmp2 = pd.DataFrame([[x[1],x[0]] if None in x else [x[0],x[1]] for x in tmp1[[1,2]].apply(list,1).to_list()])
tmp2 = tmp2.rename(columns={0:'tissue',1:'num'})
tmp2 = tmp2.assign(tissue=lambda x: x.tissue.fillna('Rectum'), num=lambda x: x.num.astype(int))
tmp3 = pd.concat([tmp1, tmp2],1)[cn_ord]
dat_cells = pd.concat([tmp3,df_cells],1)
dat_cells_long = dat_cells.melt(cn_melt,None,'cell','n')
# Get shares as well
cells = list(dat_cells_long.cell.unique())
dat_cells_long = dat_cells_long.merge(dat_cells_long.groupby(cn_melt).n.sum().reset_index().rename(columns={'n':'tot'}))
dat_cells_long = dat_cells_long.assign(share=lambda x: x.n/x.tot).drop(columns='tot')


#######################################
## --- (2) ASSIGN TRAIN/VAL/TEST --- ##

ptrain, pval = 0.8, 0.1
ptest = 1 - ptrain - pval
assert ptrain + pval + ptest == 1
share_val = pval / (pval + ptest)
print('Proportions: training (%.1f%%), validation (%.1f%%), test (%.1f%%), share (%.1f%%)' % 
    (ptrain*100, pval*100, ptest*100, share_val*100))

# Use eosinophils
dat_eosin = dat_cells_long.query('cell=="eosinophil" & ds=="hsk"').drop(columns=['cell','ds'])
dat_eosin = dat_eosin.sort_values('share').reset_index(None,True).assign(is_zero=lambda x: x.share==0)
di_is_zero = dict(zip(dat_eosin.idt_tissue, dat_eosin.is_zero))
idt_train, idt_testval = train_test_split(list(di_is_zero), train_size=ptrain, random_state=seednum, stratify=list(di_is_zero.values()))
idt_val, idt_test = train_test_split(idt_testval, train_size=share_val, random_state=seednum, stratify=[di_is_zero[z] for z in idt_testval])
# Assign the groups
lst_idt = [idt_train, idt_val, idt_test]
di_tt = dict(zip(['train','val','test'],lst_idt))
df_tt = pd.concat([pd.DataFrame({'idt_tissue':v, 'tt':k}) for k,v in di_tt.items()])
# Ensure labels are balanced
df_tt = df_tt.assign(is_zero = lambda x: [di_is_zero[z] for z in x.idt_tissue])
assert np.all(df_tt.groupby('tt').is_zero.mean().diff().abs().dropna() < 1e-2)
# Save for later
df_tt.to_csv(os.path.join(dir_output, 'train_val_test.csv'),index=False)


###############################
## --- (3) AVERAGE CELLS --- ##

tmp = dat_cells_long.merge(df_tt.drop(columns='is_zero'),'outer')
tmp.tt = tmp.tt.fillna('oos')
cell_dist = tmp.groupby(['ds','tt','cell']).n.describe()
cell_dist = cell_dist.rename(columns={'25%':'lb','75%':'ub','50%':'mu','count':'n'})
cell_dist = cell_dist.reset_index().drop(columns=['min','max','mean','std'])

cell_dist = cell_dist.assign(cell=lambda x: x.cell.str.title(), n=lambda x: x.n.astype(int),
                 ds=lambda x: x.ds.map(di_ds),
                 tt=lambda x: pd.Categorical(x.tt,list(di_tt)).map(di_tt))


posd = pn.position_dodge(0.5)
gg_cell_dist_ds = (pn.ggplot(cell_dist,pn.aes(x='tt',y='mu',color='ds')) + 
    pn.theme_bw() + pn.geom_point(position=posd) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
    pn.labs(y='Median # of cells per image') + 
    pn.ggtitle('Linerange shows IQR') + 
    pn.facet_wrap('~cell',scales='free_y',nrow=2) + 
    pn.scale_color_discrete(name='Dataset') + 
    pn.theme(subplots_adjust={'wspace': 0.15},axis_title_x=pn.element_blank()) )
gg_save('gg_cell_dist_ds.png', dir_figures, gg_cell_dist_ds, 12, 6.5)



###############################
## --- (4) CELL-SPECIFIC --- ##

# --- (i) Share/number of tissue types --- #
for cell in cells:
    for cn in ['share','n']:
        print('Cell=%s, type=%s' % (cell, cn))
        tmp = dat_cells_long.rename(columns={cn:'y'})
        tmp = tmp.query('cell==@cell').drop(columns='cell')
        assert tmp.idt_tissue.value_counts().max() == 1
        tmp = tmp.sort_values('y').reset_index(None,True)
        tmp = tmp.assign(x=lambda z: pd.Categorical(z.idt_tissue, z.idt_tissue))
        tmp_title = 'Distribution of cell counts for %s: %s' % (cell, cn)
        tmp_fn = cell + '_' + cn + '.png'
        gg_tmp = (pn.ggplot(tmp, pn.aes(x='x',y='y',fill='tissue')) + 
            pn.geom_col(color=None) + pn.ggtitle(tmp_title) + 
            pn.labs(y=cn, x='Patient/Tissue') + 
            pn.theme(axis_text_x=pn.element_blank(),axis_ticks_major_x=pn.element_blank(),
                panel_grid_major=pn.element_blank(), panel_grid_minor=pn.element_blank(),
                panel_background=pn.element_rect(fill='#ffffff')) + 
            pn.facet_wrap('~ds', scales='free_x', labeller=pn.labeller(ds=di_ds)))
        gg_save(tmp_fn, dir_labels, gg_tmp, 9, 4)
