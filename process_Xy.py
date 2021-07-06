# --- SCRIPT TO --- #

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nfill', type=int, default=1, help='How many points to pad around pixel annotation point')
parser.add_argument('--s2', type=float, default=2.0, help='Variance of Gaussian blur')
args = parser.parse_args()
nfill, s2 = args.nfill, args.s2

# # For debugging
# nfill, s2 = 1, 2.0

# number of padded points (i.e. count inflator)
fillfac = (2 * nfill + 1) ** 2
print('nfill: %i, s2: %.1f, fillfac: x%i' % (nfill, s2, fillfac))

# Load modules
import os
import hickle
import numpy as np
import pandas as pd
from PIL import Image

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from funs_support import zip_points_parse, label_blur, find_dir_cell, makeifnot
from cells import valid_cells

# Set directories
dir_base = find_dir_cell()
dir_images = os.path.join(dir_base, 'images')
dir_points = os.path.join(dir_base, 'points')
val_suffix = 'cinci'
dir_images_val = os.path.join(dir_base, 'images_'+val_suffix)
dir_points_val = os.path.join(dir_base, 'points_'+val_suffix)
dir_output = os.path.join(dir_base, 'output')
assert all([os.path.exists(ff) for ff in [dir_images, dir_points, dir_images_val, dir_points_val]])
# If first run output folders may not exist
makeifnot(dir_output)

############################
## --- (1) CHECK DATA --- ##

# Get the list of points and cropped images and makes sure they exist in both
fn_points = os.listdir(dir_points)
fn_points_val = os.listdir(dir_points_val)
tt_points = np.concatenate((np.repeat('hsk',len(fn_points)), np.repeat(val_suffix,len(fn_points_val))))
fn_images = os.listdir(dir_images)
fn_images_val = os.listdir(dir_images_val)
tt_images = np.concatenate((np.repeat('hsk',len(fn_images)), np.repeat(val_suffix,len(fn_images_val))))

dat_points = pd.DataFrame({'tt':tt_points,'points':fn_points + fn_points_val})
dat_points = dat_points.assign(fn = lambda x: x.points.str.split('\\.', expand=True).iloc[:, 0])
dat_images = pd.DataFrame({'tt':tt_images,'images':fn_images + fn_images_val})
dat_images = dat_images.assign(fn = lambda x: x.images.str.split('\\.', expand=True).iloc[:, 0])
dat_images = dat_images.query("fn.str.contains('^cleaned')").reset_index(None,True)

# Merge and check
dat_pimages = dat_points.merge(dat_images,'left',['tt','fn'])
dat_pimages.fn = dat_pimages.fn.str.replace('cleaned\\_','')
missing_points = dat_pimages[dat_pimages.isnull().any(1)]
if len(missing_points) > 0:
    print('There are %i images without any annotations: %s' % 
        (len(missing_points), missing_points.points.str.cat(sep=', ')))
    assert False

##############################
## --- (2) PROCESS DATA --- ##

# ids_tissue = dat_points.points.str.replace('cleaned\\_','',regex=True)

# Storage
di_data = dat_pimages.groupby('tt').apply(lambda x: dict(zip(x.fn,[[] for z in range(len(x.fn))])) )
di_data = di_data.to_dict()

cn_ord = ['ds','idt_tissue','cell','y','x']
tol = 2e-2
holder_err, holder_df = np.zeros([len(dat_pimages),2]), []
for ii, rr in dat_pimages.iterrows():
    tt, points, images, fn = rr['tt'], rr['points'], rr['images'], rr['fn']
    idt = fn.replace('cleaned_', '')
    print('file: %s (%i of %i)' % (idt, ii + 1, len(dat_points)))
    dir_p, dir_im = dir_points, dir_images
    if tt != 'hsk':
        dir_p = dir_p + '_' + val_suffix
        dir_im = dir_im + '_' + val_suffix        
    # (i) Load points and images
    path_points = os.path.join(dir_p, points)
    df_ii = zip_points_parse(path_points, dir_base, valid_cells)
    df_ii = df_ii.assign(idt_tissue=idt, ds=tt)[cn_ord]
    holder_df.append(df_ii)
    path_images = os.path.join(dir_im, images)
    img_vals = np.array(Image.open(path_images))

    # (ii) Apply Gaussian blur Coordinate to fill label
    # Make sure array is in height x width format
    idx_xy = df_ii[['y', 'x']].values.round(0).astype(int)
    lbls = label_blur(idx=idx_xy, cells=df_ii.cell.values, vcells=valid_cells,
                      shape=img_vals.shape[0:2], fill=nfill, s2=s2)
    est, true = np.sum(lbls) / fillfac, len(idx_xy)
    assert np.abs(est / true - 1) <= tol
    holder_err[ii] = [true, est]
    # from funs_plotting import plt_single
    # plt_single(fn='test.png',folder=dir_output,arr=img_vals,pts=lbls,thresh=1e-1)
    
    # (iii) Check cell-wise discrepancy
    cell1 = pd.DataFrame({'cell':valid_cells,'est':lbls.sum(0).sum(0)/fillfac})
    cell2 = df_ii.groupby('cell').size().reset_index().rename(columns={0:'act'})
    cell3 = cell1.merge(cell2).assign(pct=lambda x: np.abs(x.est / x.act - 1), 
        dcell=lambda x: np.abs(x.act - x.est) )
    assert np.all((cell3.pct <= tol) | (cell3.dcell <= 1))
    # (iv) Save to dictionary
    di_data[tt][idt] = {'img':img_vals, 'lbls':lbls}

# Check fillfac discrepancy
err = pd.DataFrame(holder_err,columns=['act','est']).assign(pct=lambda x: np.abs(100*(x.est/x.act-1)))
err.sort_values('pct',ascending=False).head()

# Merge the pts
df_pts = pd.concat(holder_df).reset_index(None, True)

# Make sure all images exist
assert all([[di_data[tt][idt]['img'].shape[0] > 0 for idt in di_data[tt]] for tt in di_data.keys()])

###########################
## --- (3) SAVE DATA --- ##

# --- (i) Save the location information as pandas dataframe --- #
df_cells = df_pts.pivot_table(index=['ds','idt_tissue'],columns='cell',aggfunc='size')
df_cells = df_cells.fillna(0).astype(int).reset_index()
df_cells.to_csv(os.path.join(dir_output,'df_cells.csv'),index=False)
# Save the original location data
df_pts.to_csv(os.path.join(dir_output,'df_pts.csv'),index=False)

# --- (ii) Serialize the numpy arrays --- #
print('--- Saving pickle file ---')
for ds in di_data:
    print(ds)
    path_dump = os.path.join(dir_output, 'annot_' + ds + '.pickle')
    hickle.dump(di_data[ds], path_dump, 'w')
