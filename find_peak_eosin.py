import argparse
from plotnine.facets.facet_wrap import facet_wrap

from plotnine.labels import ggtitle

parser = argparse.ArgumentParser()
parser.add_argument('--hw', type=int, help='Height/width of crop size', default=500)
parser.add_argument('--stride', type=int, help='Stride size', default=500)
parser.add_argument('--nfill', type=int, default=1, help='How many points to pad around pixel annotation point')
parser.add_argument('--hsk', default=False, action='store_true')
parser.add_argument('--cinci', default=False, action='store_true')
args = parser.parse_args()
hw, stride, nfill = args.hw, args.stride, args.nfill
hsk, cinci = args.hsk, args.cinci
print('------ hw = %i, stride = %i ------' % (hw, stride))

# # For debugging
# hw, stride, nfill = 500, 500, 1
# hsk, cinci = True, False

assert stride <= hw
fillfac = (2 * nfill + 1) ** 2

# Script to find peak eosinophil square
import os
import plotnine as pn
from PIL import Image
from time import time
from datetime import datetime
import numpy as np
import pandas as pd
from funs_support import find_dir_cell, find_dir_GI, makeifnot, read_pickle, makeifnot, zip_files
from funs_plotting import gg_save
from funs_torch import full_img_inf
import torch  # always call in torch last

dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')
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

di_ds = {}

############################
## --- (2) HSK IMAGES --- ##

if hsk:
    idt_hsk = pd.Series(os.listdir(dir_cleaned))
    di_ds['hsk'] = idt_hsk
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

if cinci:
    idt_val = pd.Series(os.listdir(dir_cleaned_val))
    di_ds['cinci'] = idt_val
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
        dtime, nleft = time() - stime, n_idt_val - (ii+1)
        rate = (ii+1) / dtime
        seta = nleft / rate
        print('~~~~~~~ ETA = %.1f minutes ~~~~~~~' % (seta/60))


################################
## --- (4) SUBSET AND ZIP --- ##

eosin_thresh = 10

df_ds = pd.DataFrame(di_ds).melt().dropna()
df_ds.rename(columns={'variable':'ds', 'value':'idt'}, inplace=True)
df_ds['idt'] = df_ds.idt.str.replace('cleaned\\_|\\.png','',regex=True)
fn_peak = pd.Series(os.listdir(dir_peak))
csv_peak = fn_peak[fn_peak.str.contains('.csv')].reset_index(None, True)
png_peak = fn_peak[fn_peak.str.contains('.png')].reset_index(None, True)
idt_peak = csv_peak.str.split('\\_',2,True)[1].str.replace('.csv','',regex=False)
tissue_peak = csv_peak.str.split('\\_',3,True)[2].fillna('Rectum')
# tissue_peak = tissue_peak.str.split('-',1,True)[0]
# tissue_peak = pd.Series(np.where(tissue_peak == 'Splenic', 'Descending', tissue_peak))
df_idt = pd.DataFrame({'fn':csv_peak, 'idt':idt_peak, 'tissue':tissue_peak}).merge(df_ds,'left')
print(df_idt.isnull().sum())

holder = []
for ii, rr in df_idt.iterrows():
    if (ii + 1) % 50 == 0:
        print(ii+1)
    fn ,idt, tissue, ds = rr
    path_idt = os.path.join(dir_peak, fn)
    tmp_df = pd.read_csv(path_idt, usecols=cells)
    tmp_df = tmp_df.assign(idt=idt, tissue=tissue, ds=ds)
    holder.append(tmp_df)
df_count = pd.concat(holder).assign(ratio=lambda x: x.eosin/(x.eosin+x.inflam))
sup_eosin = df_count.groupby(['idt','tissue']).apply(lambda x: x.loc[x.eosin.idxmax(),'eosin'])
sup_eosin = sup_eosin.reset_index().rename(columns={0:'eosin'})
sup_eosin = sup_eosin.merge(df_ds)

thresh_eosin = sup_eosin.query('eosin > @eosin_thresh').merge(df_idt,'left')
thresh_eosin['fn'] = thresh_eosin.fn.str.replace('.csv','.png',regex=False)
assert all([os.path.exists(os.path.join(dir_peak, ff)) for ff in thresh_eosin.fn])

# Save as zip
thresh_eosin.groupby('ds').apply(lambda x: zip_files(x.fn.to_list(), dir_peak, x.ds.unique()[0]+'_thresh_eosin.zip'))


###########################
## --- (5) VISUALIZE --- ##

# Distribution of the infimum
gg_sup_eosin = (pn.ggplot(sup_eosin,pn.aes(x='eosin',color='ds')) + 
    pn.theme_bw() + pn.ggtitle('Max # of eosins by image') + 
    pn.stat_ecdf() + 
    pn.scale_color_discrete(name='Dataset',labels=['Cinci','HSK']) + 
    pn.labs(y='ECDF',x='Eosinophils'))
gg_save('gg_sup_eosin.png', dir_figures, gg_sup_eosin, 6, 4)

# Individualized infimum
tmp_df = df_count.assign(gg=lambda x: x.idt + x.tissue).melt(['ds','gg'],['eosin'],'cell')
tmp_df.groupby('gg').size()
# tmp_df = tmp_df.groupby('gg').sample(frac=0.1)

lbl_ds = {'hsk':'HSK','cinci':'Cinci'}

gg_ecdf_eosin = (pn.ggplot(tmp_df,pn.aes(x='value',group='gg')) + 
    pn.theme_bw() + 
    pn.ggtitle('ECDF for eosins by image\nBlack line shows ECDF over max') + 
    pn.stat_ecdf(color='grey',alpha=0.1,size=0.5) + 
    pn.stat_ecdf(pn.aes(x='eosin'),inherit_aes=False, color='black',size=1,data=sup_eosin) + 
    pn.facet_wrap('~ds',labeller=pn.labeller(ds=lbl_ds)) + 
    pn.scale_color_discrete(name='Dataset',labels=['Cinci','HSK']) + 
    pn.labs(y='ECDF',x='Eosinophils'))
gg_save('gg_ecdf_eosin.png', dir_figures, gg_ecdf_eosin, 10, 4)

# Keep only those points where the max is at least X

