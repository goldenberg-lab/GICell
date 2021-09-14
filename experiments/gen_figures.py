# Script to provide example figures of data/labels/prediction

import os
import torch
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from cells import valid_cells, inflam_cells
from funs_support import read_pickle, zip_points_parse, find_dir_cell, get_img_range

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

###################################
## --- (2) LOAD IN THE MODEL --- ##

# Load in the "best" models for each type
fn_best = pd.Series(os.listdir(dir_best))
fn_best = fn_best.str.split('\\_',1,True)
fn_best.rename(columns={0:'cell',1:'fn'},inplace=True)
di_fn = dict(zip(fn_best.cell,fn_best.fn))
di_fn = {k:os.path.join(dir_best,k+'_'+v) for k,v in di_fn.items()}
assert all([os.path.exists(v) for v in di_fn.values()])
di_mdl = {k1: {k2:v2 for k2, v2 in read_pickle(v1).items() if k2 in ['mdl','hp']} for k1, v1 in di_fn.items()}
dat_hp = pd.concat([v['hp'].assign(cell=k) for k,v in di_mdl.items()])

# Make inference
for j in range(k):
    # Convert images into tensor for model (sample x channel x h x w)
    img = arr_images[[j]].transpose(0,3,1,2)
    img = torch.tensor(img / 255, dtype=torch.float, device=device)
    # Get inferences for each

    # Apply post-hoc

    # Get centre




##############################
## --- (3) MAKE FIGURES --- ##

# Merge all the data together
di = dict.fromkeys(fn_idt_k)

df_points.query('fn==@fn').drop(columns='fn')
for j, fn in enumerate(fn_idt_k):
    di[fn] = dict.fromkeys(['points','img','inf'])
    di[fn]['points'] = df_points.query('fn==@fn').drop(columns='fn').reset_index(None, True)
    di[fn]['img'] = arr_images[j]
    # di[fn]['inf'] = 

palette = {"eosin":"tab:green",
           "inflam":"tab:orange"}

plt.close()
nc = 3  # data, annotations, prediction
nr = k
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc*5, nr*5))
for i, ax in enumerate(axes.flatten()):
    row = i // nc
    fn = fn_idt_k[row]
    col = i % nc
    if col == 0:  # image
        ax.imshow(di[fn]['img'])
    elif col == 1:  # annotations
        sns.scatterplot(x='x', y='y', hue='cell', hue_order=list(palette), palette=palette, data=di[fn]['points'], ax=ax)
    else:
        print('Inference')
# Save output
fig.savefig(os.path.join(dir_figures,'ml_pipeline.png'))
plt.close()


