import sys, os, shutil, itertools
import pandas as pd
import numpy as np
from zipfile import ZipFile
from scipy.ndimage import gaussian_filter
import torch

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def intax3(arr):
    return np.apply_over_axes(np.sum,arr,2)

# Function to plot numpy array
# fn='both2.png'; path=dir_figures; cmap='viridis'
def array_plot(arr, path, pts=None, fn='lbl_test.png', cmap='viridis'):  #cmap='gray'
    plt.close()
    nimg = arr.shape[-1]
    dimg = len(arr.shape)
    fig, axes = plt.subplots(nimg,1,figsize=(6,6*nimg))
    vm = 1
    if arr.dtype == np.integer:
        vm = 255
    for ii, ax in enumerate(axes.flatten()):
        if dimg == 3:
            arr_ii = arr[:, :, ii]
        elif dimg == 4:
            arr_ii = arr[:, :, :, ii]
        ax.imshow(arr_ii, cmap=cmap, vmin=0, vmax=vm)
    if pts is not None:
        stopifnot(pts.shape[0:2] == arr.shape[0:2] and pts.shape[2] == arr.shape[3])
        for ii, ax in enumerate(axes.flatten()):
            pts_ii = pts[:,:,ii]
            idx_ii = np.where(pts_ii >= 0)
            ax.scatter(y=idx_ii[0], x=idx_ii[1], s=pts_ii[idx_ii]*0.1, c='orange')
    fig.savefig(os.path.join(path,fn))

# # Function to convert numpy array to torch array
# def array2torch(xx, device):
#     if len(xx.shape) == 2:  # for label
#         tens = torch.tensor(xx.transpose(2, 0, 1).astype(np.float32))
#     elif len(xx.shape) == 3:  # for image
#         tens = xx.transpose(3, 2, 0, 1).astype(np.float32)
#         tens = torch.tensor( tens / 255 )
#     return tens

# Function to convert torch array back to numpy array (vmax=255)
def torch2array(xx, vm=255):
    if len(xx.shape) == 4:  # from image
        arr = xx.cpu().detach().numpy().transpose(2, 3, 1, 0)
        arr = (arr * vm).astype(int)
    elif len(xx.shape) == 3:  # for labels
        arr = xx.cpu().detach().numpy().transpose(1, 2, 0)
    else:
        stopifnot(False)
    return arr

def stopifnot(cond):
    if not cond:
        sys.exit('error!')

# Function to create labels for each image
# idx=idx_xy.copy(); shape=img_vals.shape[0:2]; fill=1; s2=2
# cells=df_ii.cell.values; vcells=valid_cells
def label_blur(idx, cells, vcells, shape, fill=1, s2=2):
    img = np.zeros(tuple(list(shape) + [len(vcells)])) #,dtype=int
    xmx, ymx = shape[0]-1, shape[1]-1
    frange = np.arange(-fill,fill+1,1)
    nudge = np.array(list(itertools.product(frange,frange)))
    npad = nudge.shape[0]
    nc_act = np.zeros(len(vcells),int)
    for kk, cell in enumerate(vcells):
        img_kk = img[:,:,kk].copy()
        if cell in cells:
            cidx = np.where(cells == cell)[0]
            nc_act[kk] = len(cidx)
            for ii in cidx:
                x1, x2 = idx[ii, 0], idx[ii, 1]
                for jj in range(len(nudge)):
                    x1n = x1 + nudge[jj, 0]
                    x1n = max(min(x1n, xmx),0)
                    x2n = x2 + nudge[jj, 1]
                    x2n = max(min(x2n, ymx), 0)
                    img_kk[x1n, x2n] = 255
            img_kk2 = gaussian_filter(input=img_kk, sigma=2)
            deflator = np.sum(img_kk2) / np.sum(img_kk)
            img_kk2 = img_kk2 / deflator
            img[:,:,kk] = img_kk2 / 255 # / npad
        else:
            nc_act[kk] = 0
    return img

# Function to parse the zipped file
def zip_points_parse(fn, dir, valid_cells):
    valid_files = ['Points ' + str(k + 1) + '.txt' for k in range(6)]
    with ZipFile(file=fn, mode='r') as zf:
        names = pd.Series(zf.namelist())
        stopifnot(names.isin(valid_files).all())
        zf.extractall('tmp')
    # Loop through and parse files
    holder = []
    for pp in names:
        s_pp = pd.read_csv(os.path.join(dir, 'tmp', pp), sep='\t', header=None)
        stopifnot(s_pp.loc[0, 0] == 'Name')
        cell_pp = s_pp.loc[0, 1].lower()
        stopifnot(cell_pp in valid_cells)
        df_pp = pd.DataFrame(s_pp.loc[3:].values.astype(float), columns=['x', 'y'])
        stopifnot(df_pp.shape[0] == int(s_pp.loc[2, 1]))  # number of coords lines up
        df_pp.insert(0, 'cell', cell_pp)
        holder.append(df_pp)
    df = pd.concat(holder).reset_index(drop=True)
    shutil.rmtree('tmp', ignore_errors=True)  # Get rid of temporary folder
    return df
