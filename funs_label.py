# Support functions for handling the labels

# Function libraries
import os
import sys
import itertools
import shutil
import PIL
import numpy as np
import pandas as pd
from zipfile import ZipFile
from scipy.ndimage import gaussian_filter
from funs_support import vprint, drop_unnamed

# Function to create labels for each image
def label_blur(idx, cells, vcells, shape, fill=1, s2=2):
    # idx=idx_xy.copy(); shape=img_vals.shape[0:2]; fill=1; s2=2
    # cells=df_ii.cell.values; vcells=valid_cells
    img = np.zeros(tuple(list(shape) + [len(vcells)]))
    xmx, ymx = shape[0] - 1, shape[1] - 1
    frange = np.arange(-fill, fill + 1, 1)
    nudge = np.array(list(itertools.product(frange, frange)))
    npad = nudge.shape[0]
    nc_act = np.zeros(len(vcells), int)
    for kk, cell in enumerate(vcells):
        img_kk = img[:, :, kk].copy()
        if cell in cells:
            cidx = np.where(cells == cell)[0]
            nc_act[kk] = len(cidx)
            for ii in cidx:
                x1, x2 = idx[ii, 0], idx[ii, 1]
                for jj in range(len(nudge)):
                    x1n = x1 + nudge[jj, 0]
                    x1n = max(min(x1n, xmx), 0)
                    x2n = x2 + nudge[jj, 1]
                    x2n = max(min(x2n, ymx), 0)
                    img_kk[x1n, x2n] = 255
            img_kk2 = gaussian_filter(input=img_kk, sigma=2)
            deflator = np.sum(img_kk2) / np.sum(img_kk)
            img_kk2 = img_kk2 / deflator
            img[:, :, kk] = img_kk2 / 255  # / npad
        else:
            nc_act[kk] = 0
    return img



"""
Function to process the QuPath annotations
fn:             name of the file
dir:            folder where file lives
valid_cells:    list of cells
"""
def zip_points_parse(fn, dir, valid_cells,verbose=False):
    # fn=points;dir=path_points;valid_cells=valid_cells;verbose=True
    tt = fn.split('.')[-1]
    path = os.path.join(dir, fn)
    assert os.path.exists(path)
    if tt == 'tsv':
        vprint('tsv file', verbose)
        df = drop_unnamed(pd.read_csv(path, sep='\t'))
        df.rename(columns={'class':'name'},inplace=True,errors='ignore')
        assert not df['name'].isnull().all(), 'error! cell name is missing'
        if df.columns.isin(['x2','y2']).sum() == 2:
            vprint('Using rescaled columns', verbose)
            df.drop(columns=['x','y'], inplace=True)
            df.rename(columns={'x2':'x', 'y2':'y'}, inplace=True)
        df = df[['x','y','name']]
        df.rename(columns={'name':'cell'}, inplace=True)
        df['cell'] = df.cell.str.lower()
        # Remove any rows with missing cell names
        df = df[df.cell.notnull()].reset_index(None, drop=True)
        # Remove trailing "s" 
        df['cell'] = df.cell.str.replace('s$','',regex=True)
        # Remove " cell"
        df['cell'] = df.cell.str.replace('\\scell$','',regex=True)
        d_cells = pd.Series(np.setdiff1d(df.cell.unique(),valid_cells))
        assert len(d_cells) == 0, 'New cells: %s' % d_cells.str.cat(sep=', ')
    else:
        vprint('zip file', verbose)
        path_tmp = os.path.join(dir, 'tmp')
        with ZipFile(file=path, mode='r') as zf:
            zf.extractall(path_tmp)
        # Loop through and parse files
        names = pd.Series(zf.namelist())
        holder = []
        for pp in names:
            s_pp = pd.read_csv(os.path.join(path_tmp, pp), sep='\t', header=None)
            assert s_pp.loc[0, 0] == 'Name'
            cell_pp = s_pp.loc[0, 1].lower()
            assert cell_pp in valid_cells
            df_pp = pd.DataFrame(s_pp.loc[3:].values.astype(float), columns=['x', 'y'])
            assert df_pp.shape[0] == int(s_pp.loc[2, 1])  # number of coords lines up
            df_pp.insert(0, 'cell', cell_pp)
            holder.append(df_pp)
        df = pd.concat(holder).reset_index(drop=True)
        assert pd.Series(df.cell.unique()).isin(valid_cells).all()
        shutil.rmtree(path_tmp, ignore_errors=True)  # Get rid of temporary folder
    return df


"""
LOAD SPECIFIC PORTION OF PIL IMAGE IN NP.ARRAY
img:  a PIL image loaded by img=PIL.Image.open(path)
"""
# xstart, xend, ystart, yend = 10, 20, 30, 50
def get_img_range(img, xstart, xend, ystart, yend):
    assert isinstance(img,PIL.PngImagePlugin.PngImageFile)
    assert hasattr(img, 'getpixel')
    assert img.mode == 'RGB'
    h, w = img.height, img.width
    if xend+1 > w:
        print('Warning, xend is greater than width')
        xend = w-1
    if yend+1 > h:
        print('Warning, yend is greater than height')
        yend = h-1
    vals = np.zeros([yend-ystart+1,xend-xstart+1,3],dtype=int)
    for ix, x in enumerate(range(xstart, xend+1)):
        for iy, y in enumerate(range(ystart, yend+1)):
            rgb = img.getpixel(xy=(x, y))
            vals[iy,ix] = np.array(rgb)
    return vals
