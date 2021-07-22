import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hw', type=int, help='Height/width of crop size', default=500)
parser.add_argument('--stride', type=int, help='Stride size', default=500)
parser.add_argument('--nfill', type=int, default=1, help='How many points to pad around pixel annotation point')
args = parser.parse_args()
hw = args.hw
stride = args.stride
nfill = args.nfill
print('------ hw = %i, stride = %i ------' % (hw, stride))

# # For debugging
# hw, stride, nfill = 500, 500, 1

assert stride <= hw
fillfac = (2 * nfill + 1) ** 2

# Script to find peak eosinophil square
import os
from PIL import Image
from time import time
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from funs_support import find_dir_cell, find_dir_GI, makeifnot, read_pickle, makeifnot
from funs_torch import full_img_inf

dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_best = os.path.join(dir_output, 'best')
dir_peak = os.path.join(dir_output, 'peak')
makeifnot(dir_peak)
assert os.path.exists(dir_best)
dir_GI = find_dir_GI()
dir_GI_data = os.path.join(dir_GI,'data')

# Path to "cleaned" images
val_suffix = 'cinci'
dir_cleaned = os.path.join(dir_GI_data, 'cleaned')
dir_cleaned_val = os.path.join(dir_GI_data, 'cleaned' + '_' + val_suffix)
lst_cleaned = [dir_cleaned, dir_cleaned_val]
assert all([os.path.exists(dir) for dir in lst_cleaned])

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
    n_cuda = torch.cuda.device_count()
    cuda_index = list(range(n_cuda))
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
    n_cuda, cuda_index = None, None
device = torch.device('cuda:0' if use_cuda else 'cpu')

ymd = datetime.now().strftime('%Y_%m_%d')

############################
## --- (1) LOAD MODEL --- ##

cells = ['inflam', 'eosin']
cn_hp = ['lr', 'p', 'batch']

# Load in the "best" models for each typefn_best = pd.Series(os.listdir(dir_best))
fn_best = pd.Series(os.listdir(dir_best))
fn_best = fn_best.str.split('\\_',1,True)
fn_best.rename(columns={0:'cell',1:'fn'},inplace=True)
di_fn = dict(zip(fn_best.cell,fn_best.fn))
di_fn = {k:os.path.join(dir_best,k+'_'+v) for k,v in di_fn.items()}
assert all([os.path.exists(v) for v in di_fn.values()])
di_mdl = {k1: {k2:v2 for k2, v2 in read_pickle(v1).items() if k2 in ['mdl','hp']} for k1, v1 in di_fn.items()}
dat_hp = pd.concat([v['hp'].assign(cell=k) for k,v in di_mdl.items()])

print('---- BEST HYPERPARAMETERS ----')
print(dat_hp)
# Drop the hp and keep only model
di_mdl = {k: v['mdl'] for k, v in di_mdl.items()}
di_mdl = {k: v.eval() for k, v in di_mdl.items()}
di_mdl = {k: v.float() for k, v in di_mdl.items()}
di_mdl = {k: v.to(device) for k, v in di_mdl.items()}

# Models should be eval mode
assert all([not k.training for k in di_mdl.values()])


############################
## --- (2) HSK IMAGES --- ##

idt_hsk = pd.Series(os.listdir(dir_cleaned))
n_idt_hsk = len(idt_hsk)

stime = time()
for ii, idt in enumerate(idt_hsk):
    print('~~~ HSK = %s (%i of %i) ~~~' % (idt, ii+1, n_idt_hsk))
    # Get the different tissues
    dir_idt = os.path.join(dir_cleaned, idt)
    fn_idt = os.listdir(dir_idt)
    for jj, fn in enumerate(fn_idt):
        tissue = fn.split('_')[-1].replace('.png','')
        print('--- Tissue = %s (%i of %i) ---' % (tissue,jj+1,len(fn_idt)))
        path_idt = os.path.join(dir_idt, fn)
        tmp_img, tmp_inf = full_img_inf(img_path=path_idt,mdl=di_mdl,device=device,stride=stride,hw=hw)
        # Normalize by fill-factor
        tmp_inf[list(di_mdl)] = tmp_inf[list(di_mdl)].divide(fillfac)
        # Save...
        fn_img = fn.split('.')[0] + '_' + ymd + '.png'
        fn_inf = fn_img.replace('.png','.csv')
        tmp_inf.to_csv(os.path.join(dir_peak, fn_inf),index=False)
        im = Image.fromarray(tmp_img)
        im.save(os.path.join(dir_peak, fn_img))
    # ETA
    dtime, nleft = time() - stime, n_idt_hsk - (ii+1)
    rate = (ii+1) / dtime
    seta = nleft / rate
    print('ETA = %.1f minutes' % (seta/60))


##############################
## --- (3) CINCI IMAGES --- ##

idt_val = pd.Series(os.listdir(dir_cleaned_val))
n_idt_val = len(idt_val)

stime = time()
for ii, idt in enumerate(idt_val):
    print('~~~ Cinci = %s (%i of %i) ~~~' % (idt, ii+1, n_idt_val))
    # break
    path_idt = os.path.join(dir_cleaned_val, idt)
    tmp_img, tmp_inf = full_img_inf(img_path=path_idt,mdl=di_mdl,device=device,stride=stride,hw=hw)
    # Normalize by fill-factor
    tmp_inf[list(di_mdl)] = tmp_inf[list(di_mdl)].divide(fillfac)
    # Save...
    fn_img = idt.split('.')[0] + '_' + ymd + '.png'
    fn_inf = idt.replace('.png','.csv')
    tmp_inf.to_csv(os.path.join(dir_peak, fn_inf),index=False)
    im = Image.fromarray(tmp_img)
    im.save(os.path.join(dir_peak, fn_img))
    # ETA
    dtime, nleft = time() - stime, n_idt_hsk - (ii+1)
    rate = (ii+1) / dtime
    seta = nleft / rate
    print('~~~~~~~ ETA = %.1f minutes ~~~~~~~' % (seta/60))


