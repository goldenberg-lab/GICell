# Script to provide example figures of data/labels/prediction

import os
import sys
sys.path.append(os.path.join(os.getcwd()))
import torch
import numpy as np
import pandas as pd
import hickle
from PIL import Image
from cells import valid_cells, inflam_cells
from funs_support import read_pickle, zip_points_parse, find_dir_cell
from funs_inf import khat2df, phat2df, inf_thresh_cluster

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
    n_cuda = torch.cuda.device_count()
    cuda_index = list(range(n_cuda))
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
    n_cuda, cuda_index = None, None
device = torch.device('cuda:0' if use_cuda else 'cpu')

dir_base = find_dir_cell()
dir_images = os.path.join(dir_base, 'images')
dir_points = os.path.join(dir_base, 'points')
dir_output = os.path.join(dir_base, 'output')
dir_best = os.path.join(dir_output,'best')
dir_figures = os.path.join(dir_output, 'figures')

fn_points = pd.Series(os.listdir(dir_points))
fn_points = fn_points[fn_points.str.contains('png-points')]
fn_points = fn_points[~fn_points.str.contains('v2')]
fn_images = pd.Series(os.listdir(dir_images))
fn_images = fn_images[fn_images.str.contains('.png')]


##################################
## --- (1) LOAD IN THE DATA --- ##

# Find overlapping points/crops
tmp1 = fn_points.str.split('.',1,True)[0]
tmp2 = fn_images.str.split('.',1,True)[0]
fn_idt = pd.Series(np.intersect1d(tmp1, tmp2))
fn_idt = fn_idt.sort_values().reset_index(None,drop=True)

# Pick k images
k = 4
fn_idt_k = fn_idt.head(k)

holder_points = []
holder_img = []
for fn in fn_idt_k:
    # Load points
    path_points = fn_points[fn_points.str.contains(fn)].values[0]
    tmp_df = zip_points_parse(path_points, dir_points, valid_cells)
    tmp_df.insert(0,'fn',fn)
    holder_points.append(tmp_df)
    # Load images
    path_images = fn_images[fn_images.str.contains(fn)].values[0]
    path_images = os.path.join(dir_images,path_images)
    holder_img.append(np.array(Image.open(path_images)))
# Merge
df_points = pd.concat(holder_points).reset_index(None,True)
arr_images = np.stack(holder_img, axis=0)
del holder_img, holder_points
# Convert cells to inflammatory or not
df_points = df_points.assign(inflam=lambda x: x.cell.isin(inflam_cells))
df_points = df_points.query('inflam').drop(columns='inflam')
df_points = df_points.assign(cell=lambda x: np.where(x.cell == 'eosinophil','eosin','inflam'))
df_points['cell'] = pd.Categorical(df_points['cell'],['eosin','inflam'])

# Calculate image size
h, w = arr_images.shape[1:3]

###################################
## --- (2) LOAD IN THE MODEL --- ##

# (i) Load in the "best" models for each type
fn_best = pd.Series(os.listdir(dir_best))
fn_best = fn_best.str.split('\\_',1,True)
fn_best.rename(columns={0:'cell',1:'fn'},inplace=True)
di_fn = dict(zip(fn_best.cell,fn_best.fn))
di_fn = {k:os.path.join(dir_best,k+'_'+v) for k,v in di_fn.items()}
assert all([os.path.exists(v) for v in di_fn.values()])
di_mdl = {k1: {k2:v2 for k2, v2 in read_pickle(v1).items() if k2 in ['mdl','hp']} for k1, v1 in di_fn.items()}
dat_hp = pd.concat([v['hp'].assign(cell=k) for k,v in di_mdl.items()])
# Keep only model and set to inference mode
di_mdl = {k:v['mdl'].eval() for k,v in di_mdl.items()}
# Extract the module from DataParallel if exists
di_mdl = {k:v.module if hasattr(v,'module') else v for k,v in di_mdl.items()}
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
arr_phat = np.zeros([k, h, w, n_cells])
arr_yhat = arr_phat.copy()
arr_khat = arr_yhat.copy()
for j in range(k):
    print('Inference of %i of %i' % (j+1, k))
    img_j = arr_images[j]
    phat, yhat, khat = inf_thresh_cluster(mdl=di_mdl,conn=di_conn,img=img_j,device=device)
    arr_phat[j] = np.dstack(phat.values())
    arr_yhat[j] = np.dstack(yhat.values())
    arr_khat[j] = np.dstack(khat.values())


##############################
## --- (3) MAKE FIGURES --- ##

# Merge all the data together
di = dict.fromkeys(fn_idt_k)
for j, fn in enumerate(fn_idt_k):
    di[fn] = dict.fromkeys(['points','img','yhat','khat'])
    di[fn]['points'] = df_points.query('fn==@fn').drop(columns='fn').reset_index(None, drop=True)
    di[fn]['img'] = arr_images[j]
    di[fn]['phat'] = arr_phat[j]
    di[fn]['yhat'] = arr_yhat[j]
    di[fn]['khat'] = arr_khat[j]
    

# Set colors
palette = {"eosin":"tab:green", "inflam":"tab:orange"}

plt.close()
nc = 3  # ground-truth, sigmoid, cluster
nr = k
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc*5, nr*5))
for row in range(nr):
    fn = fn_idt_k[row]
    points = di[fn]['points'].copy()
    img = di[fn]['img'].copy()
    phat = di[fn]['phat'].copy()
    khat = di[fn]['khat'].copy()
    points_khat = khat2df(mat=khat, cells=cells)
    points_phat = phat2df(mat=phat, cells=cells)
    # break
    for col in range(nc):
        print('row = %i, col = %i' % (row, col))
        ax = axes[row, col]
        if col == 0:  # ground-truth
            ax.set_title('ground truth')
            ax.imshow(img,origin='lower')
            sns.scatterplot(x='x', y='y', hue='cell', hue_order=list(palette), palette=palette, data=points, ax=ax)
        elif col == 1:  # sigmoid + threshold
            ax.set_title('sigmoid + theshold')
            ax.imshow(img,origin='lower')
            ax.scatter(x=points_phat['x'],y=points_phat['y'],marker='.',s=1,linewidths=0,c=points_phat['cell'].map(palette))
        else:  # post-hoc clustering
            ax.set_title('post-hoc')
            ax.imshow(img,origin='lower')
            sns.scatterplot(x='x', y='y', hue='cell', hue_order=list(palette), palette=palette, data=points_khat, ax=ax)
    # break
# Save output
fig.savefig(os.path.join(dir_figures,'ml_pipeline.png'))
plt.close()

