"""
SCRIPT TO COMPARE FULL IMAGE RATIO TO ACTUAL ROBARTS SCORE
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-rid', '--ridx', type=int, help='Patient number to pick (0-188)', default=0)
args = parser.parse_args()
ridx = args.ridx
# ridx = 0
print('ridx = %i' % ridx)

import gc
import pickle
import os
import numpy as np
import pandas as pd
from funs_support import sigmoid, t2n, makeifnot
import torch
from plotnine import *
from PIL import Image
from funs_unet import find_bl_UNet

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device('cuda' if use_cuda else 'cpu')

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_output, 'figures')
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
dir_snapshot = os.path.join(dir_checkpoint, 'snapshot')
lst_dir = [dir_output, dir_figures, dir_checkpoint, dir_snapshot]
assert all([os.path.exists(path) for path in lst_dir])
dir_inference = os.path.join(dir_figures, 'inference')
makeifnot(dir_inference)

# Get the dates from the snapshot folder
fns_snapshot = pd.Series(os.listdir(dir_snapshot))
dates_snapshot = pd.to_datetime(fns_snapshot.str.split('\\.|\\_',5,True).iloc[:,2:5].apply(lambda x: '-'.join(x),1))
dates2 = pd.Series(dates_snapshot.sort_values(ascending=False).unique())
dnew = dates2[0].strftime('%Y_%m_%d')
print('The current date is: %s' % (dnew))
# Make folder in inference with the newest date
dir_save = os.path.join(dir_inference, dnew)
makeifnot(dir_save)

###########################################
## --- (1) LOAD DATA AND LOAD MODEL  --- ##

# cell order in the lbls matrix
valid_cells = ['eosinophil', 'neutrophil', 'plasma', 'enterocyte', 'other', 'lymphocyte']
inflam_cells = ['eosinophil', 'neutrophil', 'plasma', 'lymphocyte']
# Load data
di_img_point = pickle.load(open(os.path.join(dir_output, 'di_img_point.pickle'), 'rb'))
ids_tissue = list(di_img_point.keys())
# Image to star eosin and inflam
idx_eosin = np.where(pd.Series(valid_cells) == 'eosinophil')[0]
idx_inflam = np.where(pd.Series(valid_cells).isin(inflam_cells))[0]
for idt in ids_tissue:
    tmp = di_img_point[idt]['lbls'].copy()
    tmp_eosin = tmp[:, :, idx_eosin].sum(2)
    tmp_inflam = tmp[:, :, idx_inflam].sum(2)
    tmp2 = np.dstack([tmp_eosin, tmp_inflam])
    tmp3 = di_img_point[idt]['pts'].copy()
    tmp3 = tmp3[tmp3.cell.isin(inflam_cells)]
    di_img_point[idt]['lbls'] = tmp2
    assert np.abs( tmp3.shape[0] - (tmp_inflam.sum() / 9) ) < 1
    del tmp, tmp2, tmp3

cells = ['eosin','inflam']
# Initialize two models
fn_eosin_new, fn_inflam_new = tuple([os.path.join(dir_snapshot, 'mdl_'+cell+'_'+dnew+'.pt') for cell in cells])
mdl_eosin_new = find_bl_UNet(path=fn_eosin_new, device=device, batchnorm=True)
mdl_inflam_new = find_bl_UNet(path=fn_inflam_new, device=device, batchnorm=True)

# Load the data sources
fn_dat_new = os.path.join(dir_snapshot, 'dat_star_'+dnew+'.csv')
dat_star = pd.read_csv(os.path.join(dir_snapshot, fn_dat_new))
di_id = dat_star.groupby(['tt','id']).size().reset_index()
di_id = dict(zip(di_id.id, di_id.tt))
# Get the training/validation IDs
idt_val = [k for k,q in di_id.items() if q == 'Validation']
idt_train = [k for k,q in di_id.items() if q == 'Training']

# Create Figure with the actual ratio
df_best = dat_star.pivot_table(['act','pred'],['id','tt'],'cell').reset_index()
df_best = df_best.melt(['id','tt']).rename(columns={None:'gt'}).pivot_table('value',['id','tt','gt'],'cell').reset_index()
df_best = df_best.assign(ratio=lambda x: (x.eosin/x.inflam).fillna(0)).pivot_table('ratio',['id','tt'],'gt').reset_index()
df_best = pd.concat([dat_star.drop(columns=['epoch']),df_best.assign(cell='ratio')])

mdl_eosin_new.eval()
mdl_inflam_new.eval()
torch.cuda.empty_cache()

#########################################
## --- (2) COMPARE TO ENTIRE IMAGE --- ##

dir_GI = os.path.join(dir_base,'..','..','data')
if not os.path.exists(dir_GI):
    print('On snowqueen')
    dir_GI = os.path.join(dir_base, '..', '..', 'ordinal', 'data')
    assert os.path.exists(dir_GI)
dir_cleaned = os.path.join(dir_GI, 'cleaned')
assert os.path.exists(dir_cleaned)

kk = 1000
with torch.no_grad():
    print(mdl_eosin_new(torch.rand(1,3,kk,kk).to(device)))
    torch.cuda.empty_cache()

cn_keep = ['ID','tissue','file']
cn_nancy = ['CII','AIC']
cn_robarts = ['CII','LPN','NIE']
dat_nancy = pd.read_csv(os.path.join(dir_GI, 'df_lbls_nancy.csv'),usecols=cn_keep+cn_nancy)
dat_robarts = pd.read_csv(os.path.join(dir_GI, 'df_lbls_robarts.csv'),usecols=cn_keep+cn_robarts)
tmp_nancy = dat_nancy.melt('file',cn_nancy,'metric').assign(value=lambda x: np.where(x.value > 3, 2, x.value.fillna(0)).astype(int), tt='nancy')
tmp_robarts = dat_robarts.melt('file',cn_robarts,'metric').assign(value=lambda x: np.where(x.value > 3, 2, x.value.fillna(0)).astype(int), tt='robarts')
dat_NR = pd.concat([tmp_nancy, tmp_robarts]).reset_index(None,True)
dat_IDs = dat_nancy[['ID','tissue','file']]
del dat_nancy, dat_robarts

# Get unique file IDs
# mat_num = np.zeros([dat_IDs.shape[0],2])
# for ii, rr in dat_IDs.iterrows():
ii, rr = ridx, dat_IDs.loc[ridx]
print('Image %i of %i' % (ii+1, dat_IDs.shape[0]))
idt, tissue, file = rr['ID'], rr['tissue'], rr['file']
# Load the image
path = os.path.join(dir_cleaned,idt, file)
assert os.path.exists(path)
# img2 = cv2.imread(path, cv2.IMREAD_COLOR)
img = Image.open(path)
img = np.array(img.convert('RGB'))
height, width, channels = img.shape
print(img.shape)
# Loop over the image in convolutional chunks
nr, nd = int(np.round(width / kk)), int(np.round(height / kk))
phat_eosin = np.zeros([height, width])
phat_inflam = phat_eosin.copy()
for r in range(nr):
    for d in range(nd):
        #print('r=%i, d=%i' % (r,d))
        tmp_img = img[d * kk:(d * kk) + kk, r * kk: (r * kk) + kk].copy()
        tmp_img = np.stack([tmp_img], axis=3).transpose([3, 2, 0, 1])
        tmp_img = torch.tensor(tmp_img / 255,dtype=torch.float32).to(device)
        with torch.no_grad():
            tmp_phat_eosin = sigmoid(t2n(mdl_eosin_new(tmp_img)))
            torch.cuda.empty_cache()
            tmp_phat_inflam = sigmoid(t2n(mdl_inflam_new(tmp_img)))
            torch.cuda.empty_cache()
        phat_eosin[d * kk:(d * kk) + kk, r * kk: (r * kk) + kk] = tmp_phat_eosin
        phat_inflam[d * kk:(d * kk) + kk, r * kk: (r * kk) + kk] = tmp_phat_inflam
# # Test figure
# val_plt(img, np.dstack([phat]), np.dstack([phat]), lbls=['eosin'], path=dir_save,
#         thresh=[1e-3], fn='phat.png')
num_eosin = phat_eosin.sum()
num_inflam = phat_inflam.sum()
print(num_eosin / num_inflam)
# print(phat_eosin[phat_eosin > 1e-3].sum() / phat_inflam[phat_inflam > 1e-3].sum())
# mat_num[ii] = [num_eosin, num_inflam]
gc.collect()
del img, phat_eosin, phat_inflam, tmp_img, tmp_phat_eosin, tmp_phat_inflam
df_slice = pd.DataFrame(rr).T.assign(eosin=num_eosin, inflam=num_inflam)
fn = os.path.join(dir_save,'df_slice_'+str(ridx)+'.csv')
df_slice.to_csv(fn, index=False)

# # Merge and save
# dat_num = pd.DataFrame(mat_num, columns=['eosin','inflam'])
# dat_num = dat_num.assign(ratio=lambda x: x.eosin / (x.eosin + x.inflam))
# dat_num = pd.concat([dat_IDs, dat_num],1)
# dat_num.to_csv(os.path.join(dir_output, 'dat_cellcount.csv'),index=False)
# # Merge
# dat_NR = dat_NR.merge(dat_num[['file','ratio']])
# dat_NR.to_csv(os.path.join(dir_output, 'dat_nancyrobarts.csv'),index=False)
