# Script to explore the inter-annotator variability

import argparse

from numpy.lib.arraysetops import isin
from plotnine.facets.facet_wrap import facet_wrap
from plotnine.themes.themeable import legend_direction
parser = argparse.ArgumentParser()
parser.add_argument('-a','--annotators', nargs='+', help='List of folders for the different annotators', required=True)
parser.add_argument('--nfill', type=int, default=1, help='How many points to pad around pixel annotation point')
args = parser.parse_args()
annotators = args.annotators
nfill = args.nfill
print('args = %s' % args)

# # For debugging
# annotators = ['dua', 'oscar']
# nfill=1

# number of padded points (i.e. count inflator)
fillfac = (2 * nfill + 1) ** 2
print('nfill: %i, fillfac: x%i' % (nfill, fillfac))


import os
import torch
import hickle
import numpy as np
import pandas as pd
import plotnine as pn
from cells import valid_cells, inflam_cells
from funs_support import find_dir_cell, zip_points_parse, read_pickle
from funs_inf import khat2df, inf_thresh_cluster
from funs_plotting import gg_save
from funs_stats import get_pairwise, rho

# Set directories
dir_base = find_dir_cell()
dir_images = os.path.join(dir_base, 'images')
dir_points = os.path.join(dir_base, 'points')
dir_output = os.path.join(dir_base, 'output')
dir_best = os.path.join(dir_output,'best')
dir_figures = os.path.join(dir_output, 'figures')
assert all([os.path.exists(ff) for ff in [dir_images, dir_points, dir_output, dir_best, dir_figures]])

# Check annotators
assert all([annotator in os.listdir(dir_points) for annotator in annotators])

# Set up torch
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
    n_cuda = torch.cuda.device_count()
    cuda_index = list(range(n_cuda))
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
    n_cuda, cuda_index = None, None
device = torch.device('cuda:0' if use_cuda else 'cpu')


######################################
# --- (1) LOAD INTER-ANNOTATIONS --- #

# (i) Find overlapping images
holder = []
for annotator in annotators:
    print('Loading annotator image: %s' % annotator)
    path_anno = os.path.join(dir_output,'annot_%s.pickle' % annotator)
    assert os.path.exists(path_anno)
    tmp_di = hickle.load(path_anno)
    tmp_df = pd.DataFrame({'anno':annotator,'idt':list(tmp_di.keys())})
    holder.append(tmp_df)
dat_images = pd.concat(holder).rename_axis('idx').reset_index()
idt_images = list(dat_images.pivot('idt','anno','idx').dropna().index)
n_images = len(idt_images)
print('A total of %i images overlap' % n_images)

# (ii) Load in those specific images
img_array = np.stack([tmp_di[img]['img'] for img in idt_images],axis=0)

# (iii) Load coordinate information
cn_ord = ['anno','idt','cell','y','x']
holder = []
for annotator in annotators:
    print('Loading annotator points: %s' % annotator)
    dir_anno = os.path.join(dir_points,annotator)
    fn_anno = os.listdir(dir_anno)
    for fn in fn_anno:
        idt = fn.split('.')[0]
        if idt in idt_images:
            tmp_df = zip_points_parse(fn, dir_anno, valid_cells)
            tmp_df = tmp_df.assign(anno=annotator,idt=idt)[cn_ord]
            holder.append(tmp_df)
# Merge
df_anno = pd.concat(holder).reset_index(None,drop=True)
df_anno.rename(columns={'cell':'valid'},inplace=True)
# (iv) Calculate inflammatory/eosin/other
df_anno = df_anno.assign(cell=lambda x: np.where(x.valid.isin(inflam_cells),'inflam','other'))
df_anno = df_anno.assign(cell=lambda x: np.where(x.valid == 'eosinophil','eosin',x.cell))

# (v) Calculate aggregate
df_anno_n = df_anno.groupby(['anno','idt','cell']).size().reset_index()
df_anno_n.rename(columns={0:'n'}, inplace=True)

######################################
# --- (2) GET MODEL INFERENCES --- #


# (i) Load in the "best" models for each type
fn_best = pd.Series(os.listdir(dir_best))
fn_best = fn_best.str.split('\\_',1,True)
fn_best.rename(columns={0:'cell',1:'fn'},inplace=True)
di_fn = dict(zip(fn_best.cell,fn_best.fn))
di_fn = {k:os.path.join(dir_best,k+'_'+v) for k,v in di_fn.items()}
assert all([os.path.exists(v) for v in di_fn.values()])
di_mdl = {k: read_pickle(v)['mdl'] for k, v in di_fn.items()}
# Extract the module from DataParallel if exists
di_mdl = {k:v.module if hasattr(v,'module') else v for k,v in di_mdl.items()}
# Set to inference mode
di_mdl = {k:v['mdl'].eval() for k,v in di_mdl.items()}
# Use torch.float tensors
di_mdl = {k:v.float() for k,v in di_mdl.items()}
# Use model order for cells
cells = list(di_mdl.keys())
n_cells = len(cells)

# (ii) Extract the di_conn for optimal hyperparameter tuning
path_conn = os.path.join(dir_output,'di_conn.pickle')
di_conn = hickle.load(path_conn)
# Keep only the hyperparameters
cn_conn = ['cells','thresh','conn','n']
di_conn = {k:v for k,v in di_conn.items() if k in cn_conn}


# (iii) Get image inference
holder_n = []
holder_khat = []
for j in range(n_images):
    print('Inference of %i of %i' % (j+1, n_images))
    idt = idt_images[j]
    img_j = img_array[j]
    phat, yhat, khat = inf_thresh_cluster(mdl=di_mdl,conn=di_conn,img=img_j,device=device)
    phat = np.stack(phat.values(),2)
    khat = np.stack(khat.values(),2)
    # Get clustering count
    inf_khat = khat2df(khat, cells).drop(columns='grp')
    inf_khat = inf_khat.assign(anno='post',idt=idt)
    # Calculate aggregate number
    n_khat = inf_khat.groupby('cell').size().reset_index()
    n_khat = n_khat.rename(columns={0:'n'}).assign(idt=idt,anno='post')
    inf_phat = phat.sum(0).sum(0)/fillfac
    n_phat = pd.DataFrame({'idt':idt,'anno':'unet','cell':cells, 'n':inf_phat})
    n_j = pd.concat(objs=[n_khat, n_phat], axis=0)
    # Store
    holder_khat.append(inf_khat)
    holder_n.append(n_j)
    
# (iv) Merge annotations
anno_khat = pd.concat(holder_khat)
df_anno = pd.concat(objs=[df_anno[anno_khat.columns],anno_khat])
df_anno = df_anno[df_anno['cell'].isin(cells)].reset_index(None, drop=True)

# (v) Merge counts
df_anno_n = pd.concat(objs=[df_anno_n,pd.concat(holder_n)],axis=0)
df_anno_n = df_anno_n[df_anno_n['cell'].isin(cells)].reset_index(None, drop=True)
assert df_anno_n.groupby('anno').size().var() == 0, 'Huh?! One annotator has more images'


####################################
# --- (3) AGGREGATE STATISTICS --- #

di_anno = {'unet':'UNet', 'post':'Post-hoc', 'dua':'Dua', 'oscar':'Oscar'}
di_cell = {'eosin':'Eosinophil','inflam':'Inflammatory'}

n_bs = 250

# Get pairwise
df_pairwise_n = df_anno_n.pivot_table('n',['cell','idt'],'anno')

# (i) Get pairwise correlation and uncertainty


dat_rho_n = df_pairwise_n.groupby('cell').apply(lambda x: get_pairwise(x,rho))
dat_rho_n = dat_rho_n.reset_index().drop(columns='level_1')
dat_rho_n.rename(columns={'stat':'rho'},inplace=True)

# (ii) Plot all pairwise
holder = []
for anno in di_anno:
    tmp = df_pairwise_n.reset_index().drop(columns='idt').melt(['cell',anno]).assign(anno_x=anno)
    tmp = tmp.rename(columns={anno:'x','value':'y','anno':'anno_y'})
    holder.append(tmp)
dat_scatter = pd.concat(holder)
dat_scatter[['anno_x','anno_y']] = dat_scatter[['anno_x','anno_y']].apply(lambda x: x.map(di_anno))

gg_anno_pairwise_n = (pn.ggplot(dat_scatter,pn.aes(x='x',y='y',color='anno_y')) + 
    pn.theme_bw() + pn.labs(x='Annotator-facet',y='Annotator-color') + 
    pn.geom_point() + 
    pn.facet_wrap('~anno_x+cell',labeller=pn.labeller(cell=di_cell),scales='free',ncol=4) + 
    pn.theme(legend_position=(0.5,-0.01),legend_direction='horizontal') + 
    pn.geom_abline(intercept=0,slope=1,linetype='--') + 
    pn.theme(subplots_adjust={'wspace': 0.15,'hspace':0.30}) + 
    pn.scale_color_discrete(name='Annotator-color'))
gg_save('gg_anno_pairwise_n.png',dir_figures,gg_anno_pairwise_n,14,7)







