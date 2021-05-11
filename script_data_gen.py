###############################
## --- (0) PRELIMINARIES --- ##

import os
import shutil
import pickle
import numpy as np
import pandas as pd
from PIL import Image

from funs_support import stopifnot, zip_points_parse, label_blur, find_dir_cell

dir_base = find_dir_cell()
dir_images = os.path.join(dir_base, 'images')
dir_points = os.path.join(dir_base, 'points')
dir_output = os.path.join(dir_base, 'output')
dir_todo = os.path.join(dir_base, 'todo')
# Check
assert all([os.path.exists(ff) for ff in [dir_images, dir_points]])
# If first run, output and to do folders may not exist
for ff in [dir_output, dir_todo]:
    if not os.path.exists(ff):
        print('directory does not exist, creating: %s' % ff)
        os.mkdir(ff)

valid_cells = ['eosinophil', 'neutrophil', 'plasma',
               'enterocyte', 'other', 'lymphocyte']

fill = 1  # how many points to pad
fillfac = 2 * fill + 1  # number of padded points (i.e. count inflator)
s2 = 2  # gaussian blur

##################################
## --- (1) PREP IN THE DATA --- ##

fn_points = os.listdir(dir_points)
fn_images = os.listdir(dir_images)
raw_points = pd.Series(fn_points).str.split('\\.', expand=True).iloc[:, 0]
raw_images = pd.Series(fn_images).str.split('\\.', expand=True).iloc[:, 0]
raw_images = raw_images[raw_images.str.contains('^cleaned')].reset_index(None, True)
qq = raw_points.isin(raw_images)
stopifnot(qq.all())
print('All points found in images, %i of %i images found in points' %
      (qq.sum(), len(raw_images)))
missing_points = raw_images[~raw_images.isin(raw_points)].reset_index(None, True)
print('There are %i images without any annotations' % (len(missing_points)))

# Get who these patients are
df_missing = missing_points.str.replace('cleaned_','').str.replace('\\_[0-9]{1,3}','')
df_missing = df_missing.str.split('_',1,True).rename(columns={0:'idt',1:'tissue'})
df_missing = df_missing.sort_values(['idt','tissue']).reset_index(None,True)
# Break with codebreaker
path_breaker = os.path.join(dir_base,'..','GIOrdinal','data','df_codebreaker.csv')
df_breaker = pd.read_csv(path_breaker)
df_breaker['tissue'] = df_breaker.file2.str.split('\\_',2,True).iloc[:,2].str.replace('.png','')
df_breaker = df_breaker.drop(columns=['file','file2']).rename(columns={'QID':'idt'})
df_missing = df_missing.merge(df_breaker,on=['idt','tissue'])
df_missing.to_csv(os.path.join(dir_output,'df_missing.csv'),index=False)

# # TEMP: Compare to the send files
# fn_extra = pd.Series(os.listdir(os.path.join(dir_cell, 'archive', 'extra')))
# fn_extra = fn_extra.str.split('\\.', 1, True).iloc[:, 0]
# fn_6th = pd.Series(os.listdir(os.path.join(dir_cell, 'archive', '6th')))
# fn_6th = fn_6th.str.split('\\.', 1, True).iloc[:, 0]
# assert fn_6th.isin(fn_extra).all()
# assert fn_6th.isin(raw_points).all()
# print('A total of %i were not included file sent\n'
#       'A total of %i were not annotated in file sent' %
#       (np.sum(~missing_points.isin(fn_extra)),
#        np.sum(np.sum(fn_extra.isin(missing_points)))))

for fn in missing_points:
    fn_img = fn + '.png'
    path = os.path.join(dir_images, fn_img)
    assert os.path.exists(path)
    dest = os.path.join(dir_todo, fn_img)
    if not os.path.exists(dest):
        shutil.copy(path, dest)

ids_tissue = pd.Series(fn_points).str.replace('.png-points.zip|cleaned\\_', '')
# tmp = tmp.str.replace('\\_[0-9]{1,2}','').str.split('_',expand=True,n=1)
# ids = tmp.iloc[:,0]
# tissue = tmp.iloc[:,1]
# ids_tissue = ids + '_' + tissue

di_img_point = {z: {'img': [], 'pts': [], 'lbls': []} for z in ids_tissue}
npoint = len(fn_points)
for ii, fn in enumerate(fn_points):
    print('file %s (%i of %i)' % (fn, ii + 1, len(fn_points)))
    idt = fn.replace('.png-points.zip', '').replace('cleaned_', '')
    # idt = re.sub('\\_[0-9]{1,2}','', idt)
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
    est, true = np.sum(lbls) / fillfac ** 2, idx_xy.shape[0]
    pct = 100 * (est / true - 1)
    assert np.abs(pct) <= 2
    # from funs_support import comp_plt, val_plt
    # val_plt(arr=di_img_point[idt]['img'], pts=lbls[:,:,[0,2]],lbls=['a','b'],
    #          gt=lbls[:,:,[0,2]],path='..',thresh=[0.0,0.0],fn='blur.png')

print(len(di_img_point))
assert all([di_img_point[z]['img'].shape[0] > 0 for z in di_img_point])

# Save for later
pickle.dump(di_img_point, open(os.path.join(dir_output, 'di_img_point.pickle'), "wb"))
