# SCRIPT TO SAVE ALL LABELS AND IMAGES AS DICTIONARIES

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
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
import os
import hickle
import numpy as np
import pandas as pd
from PIL import Image
from cells import valid_cells
from funs_support import find_dir_cell, makeifnot
from funs_label import zip_points_parse, label_blur

# Set directories
dir_base = find_dir_cell()
dir_images = os.path.join(dir_base, 'images')
dir_points = os.path.join(dir_base, 'points')
assert all([os.path.exists(ff) for ff in [dir_images, dir_points]])
dir_output = os.path.join(dir_base, 'output')
makeifnot(dir_output)


############################
## --- (1) CHECK DATA --- ##

# Get the different points folders
fold_points = list(filter(lambda x: os.path.isdir(os.path.join(dir_points,x)), os.listdir(dir_points)))
fn_images = pd.Series(os.listdir(dir_images))
fn_images = fn_images[fn_images.str.contains('.png')]
idt_images = fn_images.str.replace('.png','',regex=False)
# Create data.frame for easy merge
df_images = pd.DataFrame({'idt':idt_images, 'images':fn_images})

# Loop through points folders to check for alignment
holder = []
for fold in fold_points:
    dir_fold = os.path.join(dir_points, fold)
    fn_points = pd.Series(os.listdir(dir_fold))
    idt_points = fn_points.str.split('\\.',1,True)[0]
    assert len(np.setdiff1d(idt_points, idt_images)) == 0
    tmp_df = pd.DataFrame({'tt':fold,'points':fn_points,'idt':idt_points})
    tmp_df = tmp_df.merge(df_images,'left','idt')
    holder.append(tmp_df)
# Merge for data.frame
dat_pimages = pd.concat(holder).reset_index(None,drop=True)
# Make sure no duplicates within group
assert not dat_pimages.groupby('tt').apply(lambda x: x.idt.duplicated().any()).any()
# Count
n_images = len(dat_pimages)


##############################
## --- (2) PROCESS DATA --- ##

# Storage
di_data = dat_pimages.groupby('tt').apply(lambda x: dict(zip(x.idt,[[] for z in range(len(x.idt))])) )
di_data = di_data.to_dict()

cn_ord = ['ds','idt_tissue','cell','y','x']
tol_pct, tol_dcell = 0.02, 2
holder_err = np.zeros([n_images,2])
holder_df = []
for ii, rr in dat_pimages.iterrows():
    tt, points, images, idt = rr['tt'], rr['points'], rr['images'], rr['idt']
    idt_tissue = idt.replace('cleaned_', '')
    if (ii + 1) % 25 == 0:
        print('file: %s (%i of %i)' % (idt, ii + 1, n_images))
    # (i) Load points
    path_points = os.path.join(dir_points, tt)
    df_ii = zip_points_parse(fn=points, dir=path_points, valid_cells=valid_cells)
    df_ii = df_ii.assign(idt_tissue=idt_tissue, ds=tt)[cn_ord]
    holder_df.append(df_ii.assign(ii=ii))

    # (ii) Load images
    path_images = os.path.join(dir_images, images)
    img_vals = np.array(Image.open(path_images))

    # (iii) Apply Gaussian blur
    # Make sure array is in height x width format
    idx_xy = df_ii[['y', 'x']].round(0).astype(int).values
    lbls = label_blur(idx=idx_xy, cells=df_ii.cell.values, vcells=valid_cells, shape=img_vals.shape[0:2], fill=nfill, s2=s2)
    est, true = np.sum(lbls) / fillfac, len(idx_xy)
    pct_err = np.abs(est / true - 1)
    dcell_err = np.abs(est / true)
    assert (pct_err <= tol_pct) | (dcell_err <= tol_dcell) , 'Cell discrepancy violated: %s, %i' % (idt, ii)
    holder_err[ii] = [true, est]
    
    # (iv) Check cell-wise discrepancy
    tmp1 = pd.DataFrame({'cell':valid_cells,'est':lbls.sum(0).sum(0)/fillfac})
    tmp2 = df_ii.groupby('cell').size().reset_index().rename(columns={0:'act'})
    cell_check = tmp1.merge(tmp2).assign(pct=lambda x: np.abs(x.est / x.act - 1), dcell=lambda x: np.abs(x.act - x.est) )
    assert np.all((cell_check['pct'] <= tol_pct) | (cell_check['dcell'] <= tol_dcell)), 'Cell-wise discrepancy violated: %s, %i' % (idt, ii)
    # (iv) Save to dictionary
    di_data[tt][idt] = {'img':img_vals, 'lbls':lbls}

# Check fillfac discrepancy
err = pd.DataFrame(holder_err,columns=['act','est']).assign(pct=lambda x: np.abs(100*(x.est/x.act-1)))
print(err.sort_values('pct',ascending=False).head())

# Merge the pts
df_pts = pd.concat(holder_df).reset_index(None, drop=True)

# Make sure all images exist
assert all([[di_data[tt][idt]['img'].shape[0] > 0 for idt in di_data[tt]] for tt in di_data.keys()]), 'Missing at least one image'


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
