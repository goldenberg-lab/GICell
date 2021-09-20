# Script for helpful plotting functions
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
from colormath.color_objects import BaseRGBColor
from skimage.measure import label
from matplotlib import pyplot as plt

colorz3 = np.array([sns.color_palette(None)[k] for k in [0,1,2,3]])


# --- FUNCTION TO REPRODUCE DEFAULT GGPLOT COLORS --- #

def rgb255(v):
    return np.where(v > 255, 255, np.where(v < 0, 0, v))

def cosd(d):
    r = d * np.pi / 180
    return np.cos(r)

def sind(d):
    r = d * np.pi / 180
    return np.sin(r)

def gamma_correct(u):
    GAMMA = 2.4
    if u > 0.00304:
        u = 1.055*u**(1/GAMMA) - 0.055
    else:
        u = 12.92*u
    return u
    
# Taken from: https://github.com/nickjhughes/hclmat/blob/master/hcl2rgb.m
def hcl2rgb(h, c, l):
    WHITE_Y = 100.000
    WHITE_u = 0.1978398
    WHITE_v = 0.4683363
    assert not (l < 0 or l > WHITE_Y or c < 0)
    L = l
    U = c * cosd(h)
    V = c * sind(h)
    if L <= 0 and U == 0 and V == 0:
        X = 0
        Y = 0
        Z = 0
    else:
        Y = WHITE_Y
        if L > 7.999592:
            Y = Y*((L + 16)/116)**3
        else:
            Y = Y*L/903.3
        u = U/(13*L) + WHITE_u
        v = V/(13*L) + WHITE_v
        X = (9.0*Y*u)/(4*v)
        Z = -X/3 - 5*Y + 3*Y/v
    r = gamma_correct((3.240479*X - 1.537150*Y - 0.498535*Z)/WHITE_Y)
    g = gamma_correct((-0.969256*X + 1.875992*Y + 0.041556*Z)/WHITE_Y)
    b = gamma_correct((0.055648*X - 0.204043*Y + 1.057311*Z)/WHITE_Y)
    r = rgb255(int(np.round(255 * r)))
    g = rgb255(int(np.round(255 * g)))
    b = rgb255(int(np.round(255 * b)))
    return r, g, b

def hcl2hex(h, c, l):
    r, g, b = hcl2rgb(h, c, l)
    res = BaseRGBColor(rgb_r=r, rgb_g=g, rgb_b=b, is_upscaled=True)
    hex = res.get_rgb_hex().upper()
    return hex

def gg_color_hue(n):
    hues = np.linspace(15, 375, num=n + 1)[:n]
    hcl = []
    for h in hues:
        hcl.append(hcl2hex(h=h, c=100, l=65))
    return hcl

# --- FUNCTION TO OVERWRITE EXISTING GGPLOTS --- #
def gg_save(fn,fold,gg,width,height):
    path = os.path.join(fold, fn)
    if os.path.exists(path):
        os.remove(path)
    gg.save(path, width=width, height=height)


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



# arr=img_ii.copy(); pts=phat_ii.copy(); gt=lbls_ii.copy()
# path=dir_inference; lbls=cells.copy(); thresh=di_conn['thresh'].copy(); fn=fn_idt
"""
arr: the 3-channel image
pts: the model predicted points (with len(lbls) many channels)
gt: ground truth, should be same size as pts
lbls: name for each of the channels of pts/gt
"""
def val_plt(arr, pts, gt, path, fillfac, lbls=None, thresh=1e-4, fn='some.png'):
    idt = fn.replace('.png', '')
    assert len(arr.shape) == 3
    assert len(lbls) == pts.shape[2]
    assert pts.shape == gt.shape
    nlbl = len(lbls)
    rgb_yellow = [255, 255, 0]
    fs = 16
    plt.close('all')
    fig, axes = plt.subplots(nlbl, 3, figsize=(12, 4*nlbl), squeeze=False)
    for ii in range(nlbl):
        thresh_ii = thresh[ii]
        gt_ii, pts_ii = gt[:, :, ii], pts[:, :, ii]
        idx_ii_gt = gt_ii > thresh_ii
        idx_ii_pts = pts_ii > thresh_ii
        pred, act = pts_ii.sum()/fillfac, gt_ii.sum()/fillfac
        color1 = colorz3[ii]
        # color255 = (color1 * 255).astype(int)
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



"""
PLOTTING FUNCTION TO SEE FULL PIPELINE FROM LABEL TO PHAT TO YHAT
img: the 3-channel image (h x w x 3)
lbls: ground-truth labels with Gaussian Blur (h x w x k)
phat: the probability output from the model (h x w x k)
yhat: the post-processed labels (h x w x k)
fillfac: inflation ratio for ground truth
cells: the cell names len(cells) == k
thresh: threshold to apply for phat
"""
# img=img_ii.copy(); lbls=lbls_ii.copy(); phat=phat_ii.copy(); yhat=yhat_ii.copy(); 
# fillfac=fillfac; cells=cells; thresh=di_conn['thresh'].copy();
# fold=dir_inference; fn=fn_idt; title=idt
def post_plot(img, lbls, phat, yhat, fillfac, fold, fn, cells=None, thresh=0, title=None):
    assert lbls.shape == phat.shape == yhat.shape
    assert len(img.shape) == len(lbls.shape) == len(phat.shape) == len(yhat.shape)
    h, w, k = lbls.shape
    assert (h, w) == img.shape[:2]
    assert np.all((yhat==1) | (yhat==0))

    # Set up parameters
    rgb_yellow = [255, 255, 0]
    fs = 16
    size_per = 4
    n_fig = 3
    nchannel = 3
    fig_width = n_fig * size_per
    fig_height = k * size_per
    if isinstance(thresh,float) or isinstance(thresh,int):
        thresh = np.repeat(thresh,k).astype(float)
    path = os.path.join(fold, fn)
    if os.path.exists(path):
        os.remove(path)

    # Loop over rows (cells), with if else determining the columns
    plt.close('all')
    fig, axes = plt.subplots(nrows=k, ncols=n_fig, figsize=(fig_width, fig_height), squeeze=False)
    for jj in range(k):
        thresh_jj = thresh[jj]
        lbls_jj, phat_jj, yhat_jj = lbls[:, :, jj], phat[:, :, jj], yhat[:, :, jj]
        idx_jj_lbls = lbls_jj > thresh_jj
        idx_jj_phat = phat_jj > thresh_jj
        idx_jj_yhat = yhat_jj == 1
        # Calculate the cell count
        act = lbls_jj.sum() / fillfac
        est_phat = phat_jj.sum() / fillfac
        est_yhat = label(yhat[:,:,jj],return_num=True)[1]
        color1 = colorz3[jj]
        for channel in range(nchannel):
            ax = axes[jj, channel]
            if channel == 0:  # figure
                mat = img.copy()
                mat[idx_jj_lbls] = rgb_yellow
                ax.imshow(mat, cmap='viridis', vmin=0, vmax=255)
                ax.set_title('Actual: %i' % act, fontsize=fs)
            elif channel == 1:  # phat
                mat = np.zeros(img.shape) + 1
                mat[idx_jj_phat] = color1
                mat2 = np.dstack([mat, np.sqrt(phat_jj / phat_jj.max())])
                ax.imshow(mat2, cmap='viridis', vmin=0, vmax=1)
                ax.set_title('Probability: %i' % est_phat, fontsize=fs)
            else:  # yhat
                mat = np.zeros(img.shape) + 1
                mat[idx_jj_yhat] = color1
                ax.imshow(mat, cmap='viridis', vmin=0, vmax=1)
                ax.set_title('Clustering: %i' % est_yhat, fontsize=fs)
    patches = [matplotlib.patches.Patch(color=colorz3[i], label=cells[i]) for i in range(k)]
    fig.subplots_adjust(right=0.85)
    fig.legend(handles=patches, bbox_to_anchor=(1, 0.5),fontsize=fs)
    if title is not None:
        fig.suptitle(t='ID: %s' % title, fontsize=fs, weight='bold')
    fig.savefig(path)
