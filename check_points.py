import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n','--n_hours', help="Number of hours to consider",default=24,type=int)
parser.add_argument('-nc','--n_columns', help="Number of columns for figure",default=2,type=int)
parser.add_argument('-u','--user', help="Apply to sub folder",default=None,type=str)
args = parser.parse_args()
n_hours = args.n_hours
nc = args.n_columns
user = args.user
# n_hours, nc, user = 12, 1, 'oscar'


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
from cells import valid_cells
from funs_support import find_dir_cell
from funs_label import zip_points_parse

dir_base = find_dir_cell()
dir_images = os.path.join(dir_base, 'images')
dir_points = os.path.join(dir_base, 'points')
dir_output = os.path.join(dir_base, 'output')

if not os.path.exists(dir_output):
    print('output directory does not exist, creating')
    os.mkdir(dir_output)

if user is not None:
    dir_points = os.path.join(dir_points, user)
    assert os.path.exists(dir_points)

##################################
## --- (1) LOAD IN THE DATA --- ##

fn_points = os.listdir(dir_points)
fn_images = os.listdir(dir_images)
# Find the points modified "today"
tnow = time()
nhours = [(tnow-os.path.getmtime(os.path.join(dir_points, f)))/(60**2) for f in fn_points]
fn_points = [f for f,n in zip(fn_points,nhours) if n <= n_hours]
raw_points = pd.Series(fn_points).str.split('\\.', expand=True).iloc[:, 0]
raw_images = pd.Series(fn_images).str.split('\\.', expand=True).iloc[:, 0]
assert raw_points.isin(raw_images).all()
print('The following images were annotated within %i hours: %s' %
      (n_hours,', '.join(fn_points)))
print(len(fn_points))

plt.close()
npoint = len(fn_points)
nr = (npoint // nc) + int((npoint % nc)>0)
if nr == nc == 1:
    print('Adding an extra column for flatten')
    nc = 2
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc*5, nr*5))
for ii_fn, ax in zip(enumerate(fn_points),axes.flatten()):
    ii, fn = ii_fn[0], ii_fn[1]
    print('file %s (%i of %i)' % (fn, ii + 1, len(fn_points)))
    path = os.path.join(dir_points, fn)
    df_ii = zip_points_parse(path, dir_base, valid_cells)
    df_ii.cell = pd.Categorical(df_ii.cell, categories=valid_cells)
    path_img = os.path.join(dir_images, '-'.join(fn.split('-')[:-1]))
    img_vals = np.array(Image.open(path_img))
    ax.imshow(img_vals)
    sns.scatterplot(x='x', y='y', hue='cell', data=df_ii, ax=ax)
# Save output
fn_date = 'annotated_cells_' + datetime.now().strftime('%Y_%m_%d') + '.png'
if user is not None:
    fn_date = user + '_'+ fn_date
fig.savefig(os.path.join(dir_output,fn_date))
