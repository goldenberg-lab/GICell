import sys, os, shutil, itertools
import pandas as pd
import numpy as np
from zipfile import ZipFile
from scipy.ndimage import gaussian_filter

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def intax3(arr):
    return np.apply_over_axes(np.sum, arr, 2)


def sumax3(arr):
    ss = [arr[:, :, k].sum() for k in range(arr.shape[2])]
    return np.array(ss)


def meanax3(arr):
    ss = [arr[:, :, k].mean() for k in range(arr.shape[2])]
    return np.array(ss)


def ljoin(x):
    return list(itertools.chain.from_iterable(x))


def stopifnot(cond):
    if not cond:
        sys.exit('error!')


colorz = np.array([sns.color_palette(None)[k] for k in [0, 1, 2, 5, 6, 7]])
# pts=gaussian_filter(gaussian,1).copy();arr=img.copy();gt=gaussian.copy()
# path=dir_ee;fn=id+'.png';thresh=sigmoid(b0); lbls=valid_cells
"""
arr: the 3-channel image
pts: the model predicted points (with len(lbls) many channels)
gt: ground truth, should be same size as pts
lbls: name for each of the channels of pts/gt
"""
def comp_plt(arr, pts, gt, path, lbls=None, thresh=1e-4, fn='some.png'):
    plt.close()
    # id = fn.replace('.png','')
    assert len(arr.shape) == 3
    if lbls is None:
        lbls = ['Aggergate']
    assert len(lbls) == pts.shape[2]
    assert pts.shape == gt.shape
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    pts = np.where(pts < thresh, 0, pts)
    vm = 1
    if arr.dtype == np.integer:
        vm = 255
    for ii, ax in enumerate(axes.flatten()):
        if ii == 0:
            ax.imshow(arr, cmap='viridis', vmin=0, vmax=vm)
            idx_ii = np.where(gt >= 0)
            ax.scatter(y=idx_ii[0], x=idx_ii[1], s=gt[idx_ii], c='yellow')
            ax.set_title('Ground Truth', fontsize=14)
        elif ii == 1:
            ax.imshow(arr, cmap='viridis', vmin=0, vmax=vm)
            idx_ii = np.where(pts >= 0)
            ax.scatter(y=idx_ii[0], x=idx_ii[1], s=gt[idx_ii], c='yellow')
            ax.set_title('Predicted', fontsize=14)
        elif ii == 2:
            for jj, lbl in enumerate(lbls):
                gt_jj = gt[:, :, jj].copy()
                idx_jj = np.where(gt_jj >= 0)
                ax.scatter(y=idx_jj[0], x=idx_jj[1], s=gt_jj[idx_jj], c=colorz[[jj]], label=lbl)
            ax.set_title('Ground Truth', fontsize=14)
        elif ii == 3:
            for jj, lbl in enumerate(lbls):
                pts_jj = pts[:, :, jj].copy()
                idx_jj = np.where(pts_jj >= thresh[jj])
                ax.scatter(y=idx_jj[0], x=idx_jj[1], s=pts_jj[idx_jj], c=colorz[[jj]], label=lbl)
            ax.legend(bbox_to_anchor=(1.05, 1.01))
            for ss in ax.legend_.legendHandles:
                ss._sizes = np.array([8])
            ax.set_title('Predicted', fontsize=14)
    # Set title
    act, pred = np.round(sumax3(gt) / 9, 0).astype(int), np.round(sumax3(pts) / 9, 0).astype(int)
    ap = ', '.join([str(a) + ':' + str(p) for a, p in zip(act, pred)])
    t = 'ID: %s\nActual:Predicted: %s' % (id, ap)
    fig.suptitle(t=t, fontsize=14, weight='bold')
    fig.subplots_adjust(right=0.8)
    fig.savefig(os.path.join(path, fn))
# for jj in range(nlabs):
# jj = 0
# gt_jj = gt[:,:,jj].copy()
# alpha_jj = np.where(gt_jj == 0, 0, 1)
# col_jj = np.tile(matplotlib.colors.to_rgb('yellow'),gt_jj.shape).reshape(arr.shape)
# col_jj[np.where(alpha_jj == 0)][:,2] = 1
# ax.imshow(col_jj, cmap='viridis', vmin=0, vmax=1, alpha=alpha_jj)


# Function to plot numpy array
# fn='both2.png'; path=dir_figures; cmap='viridis'
def array_plot(arr, path, pts=None, fn='lbl_test.png', cmap='viridis'):  # cmap='gray'
    plt.close()
    nimg = arr.shape[-1]
    dimg = len(arr.shape)
    fig, axes = plt.subplots(nimg, 1, figsize=(6, 6 * nimg))
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
            pts_ii = pts[:, :, ii]
            idx_ii = np.where(pts_ii >= 0)
            ax.scatter(y=idx_ii[0], x=idx_ii[1], s=pts_ii[idx_ii] * 0.1, c='orange')
    fig.savefig(os.path.join(path, fn))


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


# Function to create labels for each image
# idx=idx_xy.copy(); shape=img_vals.shape[0:2]; fill=1; s2=2
# cells=df_ii.cell.values; vcells=valid_cells
def label_blur(idx, cells, vcells, shape, fill=1, s2=2):
    img = np.zeros(tuple(list(shape) + [len(vcells)]))  # ,dtype=int
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
