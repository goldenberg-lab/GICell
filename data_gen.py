###############################
## --- (0) PRELIMINARIES --- ##

import os, pickle
import numpy as np
import pandas as pd
from PIL import Image

from support_funs_GI import stopifnot, zip_points_parse, label_blur

dir_base = os.getcwd()
dir_images = os.path.join(dir_base, '..', 'images')
dir_points = os.path.join(dir_base, '..', 'points')
dir_output = os.path.join(dir_base, '..', 'output')

if not os.path.exists(dir_output):
    print('output directory does not exist, creating')
    os.mkdir(dir_output)

valid_cells = ['eosinophil', 'neutrophil', 'plasma',
               'enterocyte', 'other', 'lymphocyte']

fill = 1 # how many points to pad
fillfac = 2*fill+1 # number of padded points (i.e. count inflator)
s2 = 2 # gaussian blur

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

tmp = pd.Series(fn_points).str.replace('.png-points.zip','').str.split('_',expand=True)
ids = tmp.iloc[:,1]
tissue = tmp.iloc[:,2]
ids_tissue = ids + '_' + tissue

di_img_point = {z: {'img':[],'pts':[], 'lbls':[]} for z in ids_tissue}
npoint = len(fn_points)
for ii, fn in enumerate(fn_points):
    print('file %s (%i of %i)' % (fn, ii + 1, len(fn_points)))
    idt = '_'.join(fn.split('_')[1:3])
    path = os.path.join(dir_points, fn)
    df_ii = zip_points_parse(path, dir_base, valid_cells)
    path_img = os.path.join(dir_images, fn.split('-')[0])
    img_vals = np.array(Image.open(path_img))
    di_img_point[idt]['pts'] = df_ii.copy()
    di_img_point[idt]['img'] = img_vals.copy()
    # Coordinate to fill label
    idx_xy = df_ii[['y', 'x']].values.round(0).astype(int)
    lbls = label_blur(idx=idx_xy, cells=df_ii.cell.values, vcells=valid_cells,
                      shape=img_vals.shape[0:2], fill=fill, s2=s2)
    di_img_point[idt]['lbls'] = lbls.copy()
    stopifnot(np.round(np.sum(lbls) / fillfac**2).astype(int) == idx_xy.shape[0])

# Save for later
pickle.dump(di_img_point, open(os.path.join(dir_output,'di_img_point.pickle'), "wb"))




