# SCRIPT TO TEST THE POINT PROCESSING #

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n','--n_hours', help="Number of hours to consider",
                    default=24,type=float)
args = parser.parse_args()
n_hours = args.n_hours

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
# ratio1: eosonphil / everything

##################################
## --- (1) LOAD IN THE DATA --- ##

fn_points = os.listdir(dir_points)
fn_images = os.listdir(dir_images)
# Find the points modified "today"
tnow = time()
nhours = [(tnow-os.path.getmtime(os.path.join(dir_points, f)))/(60**2) for f in fn_points]
fn_points = [f for f,n in zip(fn_points,nhours) if n < n_hours]
raw_points = pd.Series(fn_points).str.split('\\.', expand=True).iloc[:, 0]
raw_images = pd.Series(fn_images).str.split('\\.', expand=True).iloc[:, 0]
stopifnot(raw_points.isin(raw_images).all())
print('The following images were annotated within %0.1f hours: %s' %
      (n_hours,', '.join(fn_points)))

plt.close()
nc = 2
npoint = len(fn_points)
nr = (npoint // 2) + (npoint % 2)
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc*5, nr*5))
for ii_fn, ax in zip(enumerate(fn_points),axes.flatten()):
    ii, fn = ii_fn[0], ii_fn[1]
    print('file %s (%i of %i)' % (fn, ii + 1, len(fn_points)))
    path = os.path.join(dir_points, fn)
    df_ii = zip_points_parse(path, dir_base, valid_cells)
    df_ii.cell = pd.Categorical(df_ii.cell, categories=valid_cells)
    path_img = os.path.join(dir_images, fn.split('-')[0])
    img_vals = np.array(Image.open(path_img))
    ax.imshow(img_vals)
    sns.scatterplot(x='x', y='y', hue='cell', data=df_ii, ax=ax)
# Save output

fn_date = 'annotated_cells_' + datetime.now().strftime('%Y_%m_%d') + '.png'
fig.savefig(os.path.join(dir_output,fn_date))
