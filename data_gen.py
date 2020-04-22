###############################
## --- (0) PRELIMINARIES --- ##

import os
import numpy as np
import pandas as pd

from time import time
from datetime import datetime

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from support_funs_GI import stopifnot, zip_points_parse

dir_base = os.getcwd() # 'C:\\Users\\erik drysdale\\Documents\\projects\\GI'
dir_images = os.path.join(dir_base, '..', 'images')
dir_points = os.path.join(dir_base, '..', 'points')
dir_output = os.path.join(dir_base, '..', 'output')

if not os.path.exists(dir_output):
    print('output directory does not exist, creating')
    os.mkdir(dir_output)

valid_cells = ['eosinophil', 'neutrophil', 'plasma', 'enterocyte', 'other', 'lymphocyte']


##################################
## --- (1) PREP IN THE DATA --- ##

fn_points = os.listdir(dir_points)
fn_images = os.listdir(dir_images)
raw_points = pd.Series(fn_points).str.split('\\.', expand=True).iloc[:, 0]
raw_images = pd.Series(fn_images).str.split('\\.', expand=True).iloc[:, 0]
qq = raw_points.isin(raw_images)
stopifnot(qq.all())
print('All points found in images, %i of %i images found in points' %
      (qq.sum(), len(raw_images)))

tmp = pd.Series(fn_images).str.replace('.png','').str.split('_',expand=True)
ids = tmp.iloc[:,1]
tissue = tmp.iloc[:,2]
ids_tissue = ids + '_' + tissue

di_img_point = {z: {'img':[],'pts':[]} for z in ids_tissue}
npoint = len(fn_points)
for ii, fn in enumerate(fn_points):
    print('file %s (%i of %i)' % (fn, ii + 1, len(fn_points)))
    idt = '_'.join(fn.split('_')[1:3])
    path = os.path.join(dir_points, fn)
    df_ii = zip_points_parse(path, dir_base, valid_cells)
    #df_ii.cell = pd.Categorical(df_ii.cell, categories=valid_cells)
    path_img = os.path.join(dir_images, fn.split('-')[0])
    img_vals = np.array(Image.open(path_img))
    di_img_point[idt]['pts'] = df_ii
    di_img_point[idt]['img'] = img_vals

# https://towardsdatascience.com/objects-counting-by-estimating-a-density-map-with-convolutional-neural-networks-c01086f3b3ec





# Check image
dd = di_img_point['R9I7FYRB_Transverse']['pts'].copy()
img = di_img_point['R9I7FYRB_Transverse']['img'].copy()
cell_colors = sns.color_palette("Set2")[0:6]
di_colors = dict(zip(valid_cells, cell_colors))
plt.close()
fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.imshow(img)
for cell in valid_cells:
    idx = dd.cell == cell
    clz = np.repeat(di_colors[cell],sum(idx)).reshape([3,sum(idx)]).T
    ax.scatter(x=dd[idx].x, y=dd[idx].y, s=32, c=clz, label=cell, edgecolor='black')
ax.legend()
fig.savefig(os.path.join(dir_output,'test_cell.png'))
