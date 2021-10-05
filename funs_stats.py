import numpy as np
import torch
from funs_support import t2n
import pandas as pd
from scipy import stats
from skimage.measure import label

def rho(x, y):
    return np.corrcoef(x, y)[0,1]

def get_num_label(arr, connectivity):
    res = label(input=arr, connectivity=connectivity, return_num=True)[1]
    return res

# arr = yhat_thresh_k.copy(); connectivity=conn_k; idx=None
def lbl_freq(arr, connectivity, idx=None, ret_clust=False):
    clust_grps = label(arr, connectivity=connectivity)
    freq = pd.value_counts(clust_grps[np.nonzero(clust_grps)])
    if len(freq) == 0:
        freq = pd.DataFrame({'clust':None, 'n':0},index=[0])
    else:
        freq = freq.reset_index().rename(columns={'index':'clust',0:'n'})
    if idx is not None:
        freq.insert(0,'idx',idx)
    if ret_clust:
        return freq, clust_grps
    else:
        return freq

# # phat=phat_train[0].copy(); thresh=thresh_star; n=n_star; connectivity=conn_star
# # del phat, thresh, n, connectivity, yhat, lbls, freq, vkeep
def phat2lbl(phat, thresh, n, connectivity):
    yhat = np.where(phat >= thresh, 1, 0)
    lbls = label(input=yhat, connectivity=connectivity)
    freq = pd.value_counts(lbls[np.nonzero(lbls)])
    if len(freq) == 0:
        yhat = np.zeros(yhat.shape)
    else:
        vkeep = freq[freq >= n].index.values
        yhat = np.where(np.isin(lbls, vkeep),1,0)
    return yhat

# Function to calculate global AUROC
def global_auroc(Ytrue, Ypred):  # Ytrue, Ypred = Ybin_val.copy(), P_val.copy()
    check_YP_same(Ytrue, Ypred, ybin=True)
    n1 = np.sum(Ytrue)
    n0 = np.prod(Ytrue.shape) - n1
    den = n0 * n1
    num = sum(stats.rankdata(Ypred.flatten())[Ytrue.flatten() == 1]) - n1*(n1+1)/2
    auc = num / den
    return auc


# Function to calculate the global precision/recall curve
def global_auprc(Ytrue, Ypred, n_points=50):
    check_YP_same(Ytrue, Ypred, ybin=True)
    idx_Y = (Ytrue == 1)
    thresh_seq = np.quantile(Ypred[idx_Y], np.linspace(0, 1, n_points)[1:-1])
    holder = np.zeros([n_points-2, 3])
    for i, thresh in enumerate(thresh_seq):
        idx_thresh = Ypred > thresh
        Yhat = np.where(idx_thresh, 1, 0)
        # Presicion & recall
        prec = Ytrue[idx_thresh].mean()
        recall = Yhat[idx_Y].mean()
        holder[i] = [thresh, prec, recall]        
    df = pd.DataFrame(holder, columns=['thresh','prec','recall'])
    return df

def check_YP_same(X1, X2, ybin=False):
    assert isinstance(X1, np.ndarray) & isinstance(X2, np.ndarray)
    assert X1.shape == X2.shape
    assert (X1.min() >= 0 and X1.max() <= 1) and (X2.min() >= 0 and X2.max() <= 1)
    if ybin:
        assert np.all( (X1 == 0) | (X1 == 1))

# Calculate cross entropy between any identically sized labels and predicted probs
def cross_entropy(Ytrue, Ypred):
    check_YP_same(Ytrue, Ypred)
    # negative cross-entropy
    nce = -np.mean(Ytrue*np.log(Ypred) + (1-Ytrue)*np.log(1-Ypred))
    return nce

# Function to loop through data loader and gets labels or 
def get_YP(dataloader, model, h, w, ret_Y=False, ret_P=False):
    """
    dataloader: iterable that return (ids, lbls, images)
    di:         a dictionary that has {id:['lbls']}
    """
    assert isinstance(dataloader, torch.utils.data.dataloader.DataLoader)
    assert ret_Y != ~ret_P
    assert ret_Y + ret_P == 1
    # torch needs data formatted as n_batch * n_channel * height * width
    # reverse to height * width
    mat = np.zeros([h, w, len(dataloader)])
    if model.training == True:
        model.eval()

    jj = 0
    for idt, lbl, img in dataloader:
        assert len(idt) == 1
        if ret_P:
            with torch.no_grad():
                X = t2n(model(img))
            X = np.squeeze(X.transpose(2, 3, 1, 0))
        if ret_Y:
            X = np.squeeze(t2n(lbl).transpose(2, 3, 1, 0))
        assert X.shape == (h, w)
        mat[:,:,jj] = X
        jj += 1
    return mat
