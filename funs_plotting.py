# Script for helpful plotting functions
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
from colorspace.colorlib import HCL
from funs_support import sumax3
colorz3 = np.array([sns.color_palette(None)[k] for k in [0,1,2,3]])

# --- FUNCTION TO REPRODUCE DEFAULT GGPLOT COLORS --- #
def gg_color_hue(n):
    hues = np.linspace(15, 375, num=n + 1)[:n]
    hcl = []
    for h in hues:
        hcl.append(HCL(H=h, L=65, C=100).colors()[0])
    return hcl


# --- PRODUCE PLOT WITH IMAGES AND ANNOTATION POINTS --- #
"""
PLOT ORIGINAL FIGURE WITH ANNOTATION OVER CELL AREAS AND CORRESPONDING TRUE AREA SHOWN
"""
def plt_single(fn, folder, arr, pts, thresh=1e-2, title=None):
    plt.close()
    rgb_yellow = [255, 255, 0]
    if len(pts.shape) >= 3:
        pts = pts.sum(2)
    idx = pts > thresh
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), squeeze=False)
    for ii, ax in enumerate(axes.flat):
        if ii == 0:
            print('Original image with yellow annotation')
            mat = arr.copy()
            mat[idx] = rgb_yellow
            ax.imshow(mat)
        else:
            print('Baseline cell type')
            mat2 = np.zeros(arr.shape,dtype=int) + 255
            mat2[idx] = arr[idx].copy()
            ax.imshow(mat2)
    if title is not None:
        plt.suptitle(t=title, fontsize=10, weight='bold')
    plt.savefig(os.path.join(folder, fn))
    plt.close()


# arr=img.copy(); pts=phat.copy(); gt=gaussian.copy()
# path=dir_ee;fn=idt+'.png';thresh=sigmoid(b0); lbls=agg_cells
"""
arr: the 3-channel image
pts: the model predicted points (with len(lbls) many channels)
gt: ground truth, should be same size as pts
lbls: name for each of the channels of pts/gt
"""
def comp_plt(arr, pts, gt, path, lbls=None, thresh=1e-4, fn='some.png'):
    plt.close()
    idt = fn.replace('.png', '')
    assert len(arr.shape) == 3
    assert len(lbls) == pts.shape[2]
    assert pts.shape == gt.shape
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    rgb_yellow = [255, 255, 0]
    nlbl = len(lbls)
    vm = 1
    # Create matrices to display
    a1 = np.zeros(arr.shape, dtype=int) + 255
    a3 = np.zeros(arr.shape) + 1
    a2, a4 = a1.copy(), a3.copy()
    for kk in range(nlbl):
        thresh_kk = thresh[kk]
        idx_kk_gt = gt[:, :, kk] > thresh_kk
        idx_kk_pts = pts[:, :, kk] > thresh_kk
        for jj in range(3):  # Color channels
            a1[:, :, jj][idx_kk_gt] = rgb_yellow[jj]
            a1[:, :, jj][~idx_kk_gt] = arr[:, :, jj][~idx_kk_gt]
            a2[:, :, jj][idx_kk_pts] = rgb_yellow[jj]
            a2[:, :, jj][~idx_kk_pts] = arr[:, :, jj][~idx_kk_pts]
            a3[:, :, jj][idx_kk_gt] = colorz3[kk][jj]
            a4[:, :, jj][idx_kk_pts] = colorz3[kk][jj]
            a4[:, :, jj][~idx_kk_pts] = 1
    if arr.dtype == np.integer:
        vm = 255
    for ii, ax in enumerate(axes.flatten()):
        if ii == 0:
            ax.imshow(a1, cmap='viridis', vmin=0, vmax=vm)
            ax.set_title('Ground truth', fontsize=14)
        elif ii == 1:
            ax.imshow(a2, cmap='viridis', vmin=0, vmax=vm)
            ax.set_title('Predicted', fontsize=14)
        elif ii == 2:
            ax.imshow(a3, cmap='viridis')
            ax.set_title('Actual', fontsize=14)
        elif ii == 3:
            ax.imshow(a4, cmap='viridis')
            ax.set_title('Predicted', fontsize=14)
    patches = [matplotlib.patches.Patch(color=colorz3[i], label=lbls[i]) for i in range(nlbl)]
    fig.legend(handles=patches, bbox_to_anchor=(0.99, 0.3))
    fig.subplots_adjust(right=0.85)
    act, pred = np.round(sumax3(gt) / 9, 0).astype(int), np.round(sumax3(pts) / 9, 0).astype(int)
    ap = ', '.join([str(a) + ':' + str(p) for a, p in zip(act, pred)])
    t = 'ID: %s\nActual:Predicted: %s' % (idt, ap)
    fig.suptitle(t=t, fontsize=14, weight='bold')
    fig.savefig(os.path.join(path, fn))

# arr=img.copy(); pts=phat.copy(); gt=gt.copy()
# path=dir_save;fn='phat.png';thresh=[thresh_eosin, thresh_inflam];
# lbls=['Eosinophil','Inflammatory']
def val_plt(arr, pts, gt, path, lbls=None, thresh=1e-4, fn='some.png'):
    idt = fn.replace('.png', '')
    assert len(arr.shape) == 3
    assert len(lbls) == pts.shape[2]
    assert pts.shape == gt.shape
    nlbl = len(lbls)
    rgb_yellow = [255, 255, 0]
    vm = 1
    fs = 16
    plt.close('all')
    fig, axes = plt.subplots(nlbl, 3, figsize=(12, 4*nlbl), squeeze=False)
    for ii in range(nlbl):
        thresh_ii = thresh[ii]
        gt_ii, pts_ii = gt[:, :, ii], pts[:, :, ii]
        idx_ii_gt = gt_ii > thresh_ii
        idx_ii_pts = pts_ii > thresh_ii
        pred, act = pts_ii.sum()/9, gt_ii.sum()/9
        color1 = colorz3[ii]
        color255 = (color1 * 255).astype(int)
        for jj in range(3):
            ax = axes[ii, jj]
            if jj == 0:  # figure
                mat = arr.copy()
                mat[idx_ii_gt] = rgb_yellow
                ax.imshow(mat, cmap='viridis', vmin=0, vmax=255)
                ax.set_title('Annotations', fontsize=fs)
            elif jj == 1:  # scatter
                mat = np.zeros(arr.shape) + 1
                mat[idx_ii_gt] = color1
                ax.imshow(mat, cmap='viridis', vmin=0, vmax=1)
                ax.set_title('Actual: %i' % act, fontsize=fs)
            else:  # pred
                mat = np.zeros(arr.shape) + 1
                mat[idx_ii_pts] = color1
                mat2 = np.dstack([mat, np.sqrt(pts_ii / pts_ii.max())])
                ax.imshow(mat2, cmap='viridis', vmin=0, vmax=1)
                ax.set_title('Predicted: %i' % pred, fontsize=fs)
    patches = [matplotlib.patches.Patch(color=colorz3[i], label=lbls[i]) for i in range(nlbl)]
    fig.subplots_adjust(right=0.85)
    fig.legend(handles=patches, bbox_to_anchor=(0.5, 0.1),fontsize=fs)
    fig.suptitle(t='ID: %s' % idt, fontsize=fs, weight='bold')
    fig.savefig(os.path.join(path, fn))

# # Function to plot numpy array
# fn='test.png'; path=dir_figures; cmap='viridis'
# arr=img.copy();pts=None
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
