import sys
import os
import itertools
import pickle
import pandas as pd
import numpy as np
import zipfile
from zipfile import ZipFile
import sys
import socket

def vprint(stmt, cond=True):
    if cond:
        print(stmt)

def no_diff(x, y):
    return set(x) == set(y)

def cvec(z):
    return np.atleast_2d(z).T

def get_split(x,pat='\\s',k=0,n=5):
    return x.str.split(pat,n,True).iloc[:,k]

def t2n(x):
    return x.cpu().detach().numpy()

def makeifnot(path):
    if not os.path.exists(path):
        print('making folder')
        os.mkdir(path)

def str_subset(ss, pat, regex=True):
    if not isinstance(ss, pd.Series):
        ss = pd.Series(ss)
    ss = ss[ss.str.contains(pat,regex=regex)]
    ss.reset_index(None, drop=True, inplace=True)
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

# Function to assign different base folders for different machines
def find_dir_cell():
    cpu = socket.gethostname()
    if cpu == 'RT5362WL-GGB':
        print('On predator machine')
        di_dir = {'windows':'D:\\projects\\GICell', 'wsl':'/mnt/d/projects/GICell'}
        di_os = {'nt':'windows', 'posix':'wsl'}
        dir_cell = di_dir[di_os[os.name]]
    elif cpu == 'snowqueen':
        print('On snowqueen machine')
        dir_cell = '/data/GICell'
    elif cpu == 'cavansite':
        print('On cavansite')
        dir_cell = '/data/erik/GICell'
    elif cpu == 'malachite':
        print('On malachite')
        dir_cell = '/home/erik/projects/GICell'
    else:
        sys.exit('Where are we?!')
    return dir_cell

def find_dir_GI():
    cpu = socket.gethostname()
    if cpu == 'RT5362WL-GGB':
        print('On predator machine')
        di_dir = {'windows':'D:\\projects\\GIOrdinal', 'wsl':'/mnt/d/projects/Ordinal'}
        di_os = {'nt':'windows', 'posix':'wsl'}
        dir_GI = di_dir[di_os[os.name]]
    elif cpu == 'snowqueen':
        print('On snowqueen machine')
        dir_GI = '/data/GIOrdinal'
    elif cpu == 'cavansite':
        print('On cavansite')
        dir_GI = '/data/erik/GIOrdinal'
    elif cpu == 'malachite':
        print('On malachite')
        dir_GI = '/home/erik/projects/GIOrdinal'
    else:
        sys.exit('Where are we?!')
    return dir_GI

# Zip a list of files
def zip_files(lst, fold, zip_fn):
    tmp_base = os.getcwd()
    os.chdir(fold)
    with ZipFile(zip_fn, 'w') as zipMe:        
        for file in lst:
            zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)
    os.chdir(tmp_base)

# ---- CONVERT THE 3 HYPERPARAMETERS INTO HASH ---- #
# NOTE: returns an int
def hash_hp(df, cn):
    # df=df_slice;cn=cn_hp
    assert df.columns.isin(cn).sum() == len(cn)
    assert len(df) == 1
    df = df[cn].copy().reset_index(None,drop=True)
    df = df.loc[0].reset_index().rename(columns={'index':'hp',0:'val'})
    hp_string = pd.Series([df.apply(lambda x: x[0] + '=' + str(x[1]), 1).str.cat(sep='_')])
    code_hash = pd.util.hash_pandas_object(hp_string).values[0]
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

# Function to convert torch array back to numpy array (vmax=255)
def torch2array(xx, vm=255):
    d_shape = len(xx.shape)
    assert (d_shape == 3) | (d_shape == 4), 'xx is not 3d or 4d'
    if d_shape == 4:  # from image
        arr = xx.cpu().detach().numpy().transpose(2, 3, 1, 0)
        arr = (arr * vm).astype(int)
    else:
        arr = xx.cpu().detach().numpy().transpose(1, 2, 0)
    return arr

# Remove columns with "Unnamed" from pandas df
def drop_unnamed(x):
    assert isinstance(x, pd.DataFrame)
    cn = x.columns
    cn_drop = list(cn[cn.str.contains('Unnamed')])
    if len(cn_drop) > 0:
        x = x.drop(columns=cn_drop)
    return x
