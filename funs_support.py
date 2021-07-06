import sys
import os
import shutil
import itertools
import pickle
import pandas as pd
import numpy as np
from zipfile import ZipFile
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error as mse
from arch.bootstrap import IIDBootstrap

import sys
import socket

def find_dir_cell():
    cpu = socket.gethostname()
    # Set directory based on CPU name
    if cpu == 'RT5362WL-GGB':
        print('On predator machine')
        if os.name == 'nt':
            print('We are on Windows')
            dir_cell = 'D:\\projects\\GICell'
        elif os.name == 'posix':
            dir_cell = '/mnt/d/projects/GICell'
        else:
            assert False
    elif cpu == 'snowqueen':
        print('On snowqueen machine')
        dir_cell = '/data/GICell'
    elif cpu == 'cavansite':
        print('On cavansite')
        dir_cell = '/data/erik/GICell/'
    else:
        sys.exit('Where are we?!')
    return dir_cell

# ---- CONVERT THE 3 HYPERPARAMETERS INTO HASH ---- #
# NOTE: returns an int
def hash_hp(df, method='hash_array'):
    cn_hp = ['lr', 'p', 'batch']
    assert hasattr(pd.util, method)
    fun = getattr(pd.util, method)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.isin(cn_hp).sum() == len(cn_hp)
    assert len(df) == 1
    df = df[cn_hp].copy().reset_index(None,True)
    df = df.loc[0].reset_index().rename(columns={'index':'hp',0:'val'})
    hp_string = pd.Series([df.apply(lambda x: x[0] + '=' + str(x[1]), 1).str.cat(sep='_')])
    code_hash = fun(hp_string)[0]
    return code_hash

# ---- FUNCTIONS TO READ/WRITE PICKLES ---- #
def read_pickle(path):
    assert os.path.exists(path)
    with open(path, 'rb') as handle:
        di = pickle.load(handle)
    return di

def write_pickle(di, path):
    with open(path, 'wb') as handle:
        pickle.dump(di, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ---- Index come column of df (cn_val) to smallest point (cn_idx) --- #
def idx_first(df, cn_gg, cn_idx, cn_val):
  if isinstance(cn_gg, str):
    cn_gg = [cn_gg]
  assert isinstance(cn_idx, str)
  assert isinstance(cn_val, str)
  df = df.copy()
  cn_val_min = cn_val + '_mi'
  idx_min = df.groupby(cn_gg).apply(lambda x: x[cn_idx].idxmin())
  idx_min = idx_min.reset_index().rename(columns={0:'idx'})
  val_min = df.loc[idx_min.idx,cn_gg + [cn_val]]
  val_min.rename(columns={cn_val:cn_val_min}, inplace=True)
  df = df.merge(val_min,'left',cn_gg)
  df = df.assign(val_idx = lambda x: x[cn_val]/x[cn_val_min]*100).drop(columns=[cn_val_min])
  df = df.drop(columns=cn_val).rename(columns={'val_idx':cn_val})
  return df


def no_diff(x, y):
    return set(x) == set(y)

def cvec(z):
    return np.atleast_2d(z).T

def norm_mse(y,yhat):
    mu, se = y.mean(), y.std()
    y_til, yhat_til = (y-mu)/se, (yhat-mu)/se
    return mse(y_til, yhat_til)


def get_split(x,pat='\\s',k=0,n=5):
    return x.str.split(pat,n,True).iloc[:,k]

def t2n(x):
    return x.cpu().detach().numpy()


def makeifnot(path):
    if not os.path.exists(path):
        print('making folder')
        os.mkdir(path)


def str_subset(ss, pat):
    if not isinstance(ss, pd.Series):
        ss = pd.Series(ss)
    ss = ss[ss.str.contains(pat)].reset_index(None, True)
    return ss


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

def quantax3(arr,q=0.5):
    ss = [np.quantile(arr[:, :, k],q) for k in range(arr.shape[2])]
    return np.array(ss)


def ljoin(x):
    return list(itertools.chain.from_iterable(x))


def stopifnot(cond):
    if not cond:
        sys.exit('error!')

def bootstrap_metric(act, pred, metric, nbs=999):
    ci = IIDBootstrap(act, pred).conf_int(metric, reps=nbs, method='bca', size=0.95, tail='two').flatten()
    return ci[0], ci[1]

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
# fn=path; dir=dir_base
def zip_points_parse(fn, dir, valid_cells):
    tt = fn.split('.')[-1]
    if tt == 'tsv':
        print('tsv file')
        df = pd.read_csv(fn, sep='\t', usecols=['x','y','name'])
        df.rename(columns={'name':'cell'}, inplace=True)
        df.cell = df.cell.str.lower()
        # Remove any rows with missing cell names
        df = df[df.cell.notnull()].reset_index(None,True)
        d_cells = np.setdiff1d(df.cell.unique(),valid_cells)
        if len(d_cells) > 0:
            print('New cells: %s' % (d_cells.join(', ')))
            sys.exit('Unidentified cell')
    else:
        print('zip file')
        path_tmp = os.path.join(dir, 'tmp')
        with ZipFile(file=fn, mode='r') as zf:
            zf.extractall(path_tmp)
        # Loop through and parse files
        names = pd.Series(zf.namelist())
        # valid_files = ['Points ' + str(k + 1) + '.txt' for k in range(7)]
        # if not names.isin(valid_files).all():
        #     stopifnot(False)
        holder = []
        for pp in names:
            s_pp = pd.read_csv(os.path.join(path_tmp, pp), sep='\t', header=None)
            stopifnot(s_pp.loc[0, 0] == 'Name')
            cell_pp = s_pp.loc[0, 1].lower()
            stopifnot(cell_pp in valid_cells)
            df_pp = pd.DataFrame(s_pp.loc[3:].values.astype(float), columns=['x', 'y'])
            stopifnot(df_pp.shape[0] == int(s_pp.loc[2, 1]))  # number of coords lines up
            df_pp.insert(0, 'cell', cell_pp)
            holder.append(df_pp)
        df = pd.concat(holder).reset_index(drop=True)
        assert pd.Series(df.cell.unique()).isin(valid_cells).all()
        shutil.rmtree('tmp', ignore_errors=True)  # Get rid of temporary folder
    return df
