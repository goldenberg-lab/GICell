import torch
import numpy as np
import pandas as pd
from scipy import stats
from skimage.measure import label
from arch.bootstrap import IIDBootstrap
from sklearn.metrics import mean_squared_error as mse
from funs_support import t2n

# Split dataset by continuous value
def stratify_continuous(group, score, percent, seed=None):
    if not isinstance(group, pd.Series):
        group = pd.Series(group)
    assert not group.duplicated().any()
    n = len(score)
    n_split = int(n * percent)
    df = pd.DataFrame({'score':np.sort(score)})
    df = df.rename_axis('idx').reset_index()
    bins = np.linspace(0, n, n_split+1)
    df['gg'] = pd.cut(df['idx'],bins,right=False)
    if seed is not None:
        np.random.seed(seed)
    idx_test = df.groupby('gg').sample(n=1).idx.values
    idx_train = np.setdiff1d(df['idx'], idx_test)
    di = {'test':idx_test, 'train':idx_train}
    return di


# MSE on normalized values
def norm_mse(y,yhat):
    mu, se = y.mean(), y.std()
    y_til, yhat_til = (y-mu)/se, (yhat-mu)/se
    return mse(y_til, yhat_til)

# Wrapper to get spearman correlation
def rho(x, y):
    return np.corrcoef(x, y)[0,1]

# Use skimage's clustering approach to count number of labels
def get_num_label(arr, connectivity):
    res = label(input=arr, connectivity=connectivity, return_num=True)[1]
    return res

# Get all pairwise measures for some stat (function)
def get_pairwise(df, stat, lower=True):
    # df=dat_rho_n.query('cell=="eosin"').reset_index(None,drop=True)
    # stat=r2_score;lower=True
    assert isinstance(df, pd.DataFrame)
    cn = list(df.columns)
    n_cn = len(cn)
    holder = []
    if lower:
        for i in range(n_cn-1):
            cn_i = cn[i]
            for j in range(i+1,n_cn):
                cn_j = cn[j]
                stat_ij = stat(df[cn_i],df[cn_j])
                tmp = pd.DataFrame({'cn_1':cn_i,'cn_2':cn_j,'stat':stat_ij},index=[0])
                holder.append(tmp)
    else:
        for i in range(n_cn):
            cn_i = cn[i]
            for j in range(n_cn):
                cn_j = cn[j]
                stat_ij = stat(df[cn_i],df[cn_j])
                tmp = pd.DataFrame({'cn_1':cn_i,'cn_2':cn_j,'stat':stat_ij},index=[0])
                holder.append(tmp)
    res = pd.concat(holder).reset_index(None,drop=True)
    res.insert(0,'fun',stat.__name__)
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
def global_auroc(Ytrue, Ypred):  
    # Ytrue, Ypred = bin_Yval.copy(), P_val.copy()
    check_YP_same(Ytrue, Ypred, ybin=True)
    n1 = int(np.nansum(Ytrue))
    n0 = int(np.nansum(1 - Ytrue)) - n1
    den = n0 * n1
    flat_pred = Ypred.flatten()
    flat_y = Ytrue.flatten()
    idx_keep = ~np.isnan(flat_y)
    flat_y = flat_y[idx_keep]
    flat_pred = flat_pred[idx_keep]
    num = sum(stats.rankdata(flat_pred)[flat_y == 1]) - n1*(n1+1)/2
    auc = num / den
    return auc


# Function to calculate the global precision/recall curve
def global_auprc(Ytrue, Ypred, n_points=50):
    # Ytrue=bin_Yval.copy();Ypred=Phat_init.copy();n_points=50
    check_YP_same(Ytrue, Ypred, ybin=True)
    flat_pred = Ypred.flatten()
    flat_y = Ytrue.flatten()
    idx_keep = ~np.isnan(flat_y)
    flat_y = flat_y[idx_keep]
    flat_pred = flat_pred[idx_keep]
    idx_Y1 = (flat_y == 1)
    thresh_seq = np.quantile(flat_pred[idx_Y1], np.linspace(0, 1, n_points)[1:-1])
    holder = np.zeros([n_points-2, 3])
    for i, thresh in enumerate(thresh_seq):
        idx_thresh = flat_pred > thresh
        flat_yhat = np.where(idx_thresh, 1, 0)
        # Presicion & recall
        prec = flat_y[idx_thresh].mean()
        recall = flat_yhat[idx_Y1].mean()
        holder[i] = [thresh, prec, recall]
    df = pd.DataFrame(holder, columns=['thresh','prec','recall'])
    return df

def check_YP_same(X1, X2, ybin=False):
    assert isinstance(X1, np.ndarray) & isinstance(X2, np.ndarray)
    assert X1.shape == X2.shape
    x1_min, x1_max = np.nanmin(X1), np.nanmax(X1)
    x2_min, x2_max = np.nanmin(X2), np.nanmax(X2)
    assert (x1_min >= 0 and x1_max <= 1) and (x2_min >= 0 and x2_max <= 1)
    if ybin:
        assert np.all(np.isin(np.unique(X1[~np.isnan(X1)]),[0,1])), 'Not all ybinary values are 0/1'

# Calculate cross entropy between any identically sized labels and predicted probs
def cross_entropy(Ytrue, Ypred):
    # Ytrue=bin_Yval.copy();Ypred=P_val.copy()
    check_YP_same(Ytrue, Ypred)
    flat_pred = Ypred.flatten()
    flat_y = Ytrue.flatten()
    idx_keep = ~np.isnan(flat_y)
    flat_y = flat_y[idx_keep]
    flat_pred = flat_pred[idx_keep]
    # negative cross-entropy
    nce = -np.mean(flat_y*np.log(flat_pred) + (1-flat_y)*np.log(1-flat_pred))
    return nce

# Function to loop through data loader and gets labels and images into single 
def get_YP(dataloader, model, ret_Y=False, ret_P=False, ret_idx=False):
    # dataloader=val_gen;ret_Y=True;ret_P=False;ret_idx=True;model=mdl
    """
    dataloader: iterable that return (ds, ids, lbls, images)
    """
    assert isinstance(dataloader, torch.utils.data.dataloader.DataLoader), 'dataloader is not the expected torch class'
    assert ret_Y != ~ret_P
    assert ret_Y + ret_P == 1
    # (i) Loop over data loader to calculate the max pixels
    n_batch = len(dataloader)
    h_max, w_max = 0, 0
    df_idx = pd.DataFrame(np.zeros([n_batch, 2]), columns = ['ds','fn'])
    jj = 0
    for ds, fn, lbl, img in dataloader:
        assert img.shape[2:] == lbl.shape[2:], 'img and lbl shape do not align...'
        h, w = img.shape[2:]
        h_max = max(h, h_max)
        w_max = max(w, w_max)
        df_idx.loc[jj] = [ds[0], fn[0]]
        jj += 1
        
    # (ii) Loop over and store either prediction or label
    mat = np.zeros([n_batch, h_max, w_max])
    if model.training == True:
        model.eval()

    jj = 0
    for _, _, lbl, img in dataloader:
        if ret_P:
            with torch.no_grad():
                X = np.squeeze(t2n(model(img)))
        if ret_Y:
            X = np.squeeze(t2n(lbl))
        # Assign from top-left
        h, w = X.shape
        mat[jj][:h, :w] = X
        mat[jj][:h, w:] = np.nan
        mat[jj][h:, :w] = np.nan
        mat[jj][h:, w:] = np.nan
        jj += 1

    if ret_idx:
        return df_idx, mat
    else:
        return mat


# Performs BCA bootstrap on some function
def bootstrap_metric(act, pred, metric, nbs=999):
    ci = IIDBootstrap(act, pred).conf_int(metric, reps=nbs, method='bca', size=0.95, tail='two').flatten()
    return ci[0], ci[1]

# Performs jackknife on some metric
def jackknife_metric(act, pred, metric):
    assert len(act) == len(pred)
    if isinstance(act, pd.Series):
        act = act.values
    if isinstance(pred, pd.Series):
        pred = pred.values
    n = len(act)
    vec = np.zeros(n)
    r2 = metric( act , pred )
    for ii in range(n):
        vec[ii] = metric(np.delete(act, ii), np.delete(pred, ii))
    mi, mx = min(vec), max(vec)
    bias = r2 - np.mean(vec)
    mi, mx = mi + bias, mx + bias
    return mi, mx
