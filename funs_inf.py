# Function support to do model inference with trained images

import gc
import PIL
import torch
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label
from funs_stats import lbl_freq
from funs_support import sigmoid, t2n


# Prevent overflow
PIL.Image.MAX_IMAGE_PIXELS = 1933120000


"""
function that takes phat.clip(thresh,p) and returns x/y locations
mat:        A h x w x n_cell matrix of predictions
cells:      a len(n_cell) array of cell names
"""
# mat=phat.copy()
def phat2df(mat, cells):
    assert isinstance(mat, np.ndarray), 'mat is not a numpy array'
    assert len(mat.shape) == 3, 'mat is not 3d'
    h, w, n_cells = mat.shape
    assert n_cells == len(cells), 'len(cells) != n_cells'
    idx_cells = dict(zip(range(n_cells),cells))
    df = pd.DataFrame(np.stack(np.where(mat != 0)).T,columns=['y','x','cell'])
    df['cell'] = df['cell'].map(idx_cells)
    df = df.sort_values(['cell','y','x']).reset_index(None,drop=True)
    return df


"""
function that takes label-clust and return x/y or locations
mat:        A h x w x n_cell matrix of predictions
cells:      a len(n_cell) array of cell names
"""
# mat=yhat.copy()
def khat2df(mat, cells):
    assert isinstance(mat, np.ndarray), 'mat is not a numpy array'
    assert len(mat.shape) == 3, 'mat is not 3d'
    h, w, n_cells = mat.shape
    assert n_cells == len(cells), 'len(cells) != n_cells'
    holder = []
    for j, cell in enumerate(cells):
        # Get connections
        lbl_j = label(input=mat[:,:,j],connectivity=2)
        grps = np.setdiff1d(np.unique(lbl_j),0)
        for grp in grps:
            tmp_df = pd.DataFrame(np.vstack(np.where(lbl_j == grp)).T,columns=['y','x'])
            tmp_df.insert(0,'grp',grp)
            tmp_df.insert(0,'cell',cell)
            holder.append(tmp_df)
    # Calculate the centroid of each point
    res = pd.concat(holder).groupby(['cell','grp'])[['y','x']].mean().reset_index()
    return res


"""
Function to get inference on trained model with calibrated threshold and cluster
mdl:        A dictionary by different cells
conn:       A dictionary with thresh, conn, and n keys
img:        A h x w x c numpy array
"""
# mdl=di_mdl.copy();conn=di_conn.copy();img=arr_images[j].copy()
def inf_thresh_cluster(mdl, conn, img, device):
    # (i) Run checks
    assert isinstance(mdl, dict), 'mdl is not a dictionary'
    assert isinstance(conn, dict), 'conn is not a dictionary'
    assert all([key in list(conn.keys()) for key in ['thresh', 'conn', 'n']]), 'conn is missing required keys'
    assert isinstance(img, np.ndarray), 'img is not a numpy array'
    assert len(img.shape) == 3, 'img is not 3-dimensional'
    assert isinstance(device, torch.device), 'device is not a torch.device'
    
    # (ii) Transform image to torch
    if img.max() > 1:
        img = img.astype(float) / 255
        assert img.max() <= 1, 'Huh, more than 255 for RGB?'
    img = np.expand_dims(img.transpose(2,0,1),0)
    timg = torch.tensor(img, device=device, dtype=torch.float)
    
    # (iii) Run image through model
    phat = {k: np.squeeze(sigmoid(t2n(v(timg)))) for k,v in mdl.items()}
    
    # (iv) Apply thresholding and clustering
    yhat = {}
    khat = {}
    for k in mdl.keys():
        idx_k = np.where(np.array(conn['cells']) == k)[0]
        thresh_k = conn['thresh'][idx_k][0]
        conn_k = int(conn['conn'][idx_k][0])
        n_k = conn['n'][idx_k][0]
        phat_thresh_k = np.where(phat[k] >= thresh_k, phat[k], 0.0)
        yhat_thresh_k = np.where(phat[k] >= thresh_k, 1, 0).astype(int)
        freq_k, clust_k = lbl_freq(yhat_thresh_k, conn_k, ret_clust=True)
        freq_k = freq_k[freq_k['n'] >= n_k]['clust'].values
        clust_k = np.where(np.isin(clust_k,freq_k),clust_k,0).astype(int)
        # Save
        phat[k] = phat_thresh_k
        yhat[k] = yhat_thresh_k
        khat[k] = clust_k

    # (v) Return
    return phat, yhat, khat


"""
Function to do efficient inference on image that is too big to hold in memory
img_path: path to .png
mdl: UNet or dictionary of UNet models
stride, hw: stride, height+width
returns img and dataframe
"""
# img_path = path_idt; mdl=di_mdl; stride=250; hw=500
def full_img_inf(img_path, mdl, device, stride=500, hw=500):
    assert isinstance(mdl, dict)
    assert 'eosin' in mdl
    # Load the image
    img = Image.open(img_path)
    img = img.convert('RGB')
    # THIS PART HERE CAN BE MODIFIED # 
    img = np.array(img)
    height, width, channels = img.shape
    print('Image dimensions = %s' % (img.shape,))
    # Loop over the image in convolutional chunks
    right_stride, right_rem = divmod(width - hw, stride)
    down_stride, down_rem = divmod(height - hw, stride)
    right_stride += 1
    down_stride += 1
    right_stride += int(right_rem>0)
    down_stride += int(down_rem>0)
    holder = []
    for r in range(right_stride):
        for d in range(down_stride):
            # Get image location
            ylo = stride*d
            yup = min(hw + stride*d, height)
            xlo = stride*r
            xup = min(hw + stride*r, width)
            if yup == height:
                ylo = yup - hw
            if xup == width:
                xlo = xup - hw
            print('Convolution: r=%i, d=%i (y=%i:%i, x=%i:%i)' % (r,d,ylo,yup,xlo,xup))
            # Convert image to tensor
            tmp_img = img[ylo:yup, xlo:xup]
            tmp_img = np.expand_dims(tmp_img.transpose([2,0,1]),0)
            tmp_img = torch.tensor(tmp_img / 255, dtype=torch.float32).to(device)
            with torch.no_grad():
                tmp_di = {k:np.sum(sigmoid(t2n(v(tmp_img)))) for k,v in mdl.items()}
            torch.cuda.empty_cache()
            tmp_di = {k:[v] for k,v in tmp_di.items()}
            tmp_df = pd.DataFrame.from_dict(tmp_di)
            tmp_df = tmp_df.assign(xlo=xlo,xup=xup,ylo=ylo,yup=yup)
            holder.append(tmp_df)
    # Merge
    res_inf = pd.concat(holder).reset_index(None, True)
    # Get the "best"
    xlo, xup, ylo, yup = res_inf.loc[res_inf.eosin.idxmax(),['xlo','xup','ylo','yup']].astype(int)
    img_star = img[ylo:yup,xlo:xup]
    del img
    gc.collect()
    return img_star, res_inf
