# Find out which of the the possible images have received an annotation

import os
import pandas as pd
import numpy as np
from funs_support import find_dir_cell

dir_base = find_dir_cell()
dir_images = os.path.join(dir_base, 'images')
dir_points = os.path.join(dir_base, 'points')
dir_output = os.path.join(dir_base, 'output')
dir_peak = os.path.join(dir_output, 'peak')
dir_ordinal = os.path.join(dir_base,'..','GIOrdinal','data')
dir_20x = os.path.join(dir_ordinal,'20X')
# Load in the code breaker
cn_breaker = ['ID','tissue','type','QID']
path_breaker = os.path.join(dir_ordinal,'df_codebreaker.csv')
df_breaker = pd.read_csv(path_breaker,usecols=cn_breaker)
df_breaker.rename(columns={'QID':'idt','type':'tt','ID':'oid'},inplace=True)
df_breaker = df_breaker.drop_duplicates().reset_index(level=None, drop=True)
df_nancy = pd.read_csv(os.path.join(dir_ordinal,'df_lbls_nancy.csv'))

# Get the "new" images
fn_20x = pd.Series(os.listdir(dir_20x)).str.split('\\s',1,True)[0]
fn_20x = fn_20x[fn_20x.str.contains('^S')].reset_index(level=None, drop=True)
# Note that "SH19-1002" was a failed image so removing
fn_20x = pd.Series(np.setdiff1d(fn_20x,['SH19-1002']))

# Get the image list
fn_images = pd.Series(os.listdir(dir_images))
fn_images = fn_images[fn_images.str.contains('png$')].reset_index(level=None, drop=True)

idt_images = fn_images.str.split('\\_',2,True).drop(columns=[0])
idt_images = pd.concat(objs=[idt_images[[1]], idt_images[2].str.replace('.png','',regex=True).str.split('\\_',1,True)[[0]]],axis=1)
idt_images.rename(columns={1:'idt',0:'tissue'},inplace=True)
idt_images = idt_images.drop_duplicates().reset_index(level=None, drop=True)
# Check that codebreaker lines up
cn_merge = ['idt','tissue']
assert idt_images.merge(df_breaker,'left',cn_merge).notnull().all().all()

# Get the points list
fn_points = pd.Series(os.listdir(dir_points)).str.replace('cleaned_','').str.split('.',1,True)[0]
idt_points = fn_points.str.split('\\_',2,True).rename(columns={0:'idt',1:'tissue'}).drop(columns=[2])
idt_points = idt_points.groupby(['idt','tissue']).size().reset_index().rename(columns={0:'n_anno'}).assign(is_anno=True)

# Merge all
idt_merge = idt_images.merge(idt_points,'left',cn_merge)
idt_merge.is_anno = idt_merge.is_anno.fillna(False)
idt_merge = idt_merge.merge(df_breaker,'left',cn_merge)
idt_merge = idt_merge.assign(is_new=lambda x: x.oid.isin(fn_20x) & (x.tissue=='Rectum'),n_anno=lambda x: x.n_anno.fillna(0).astype(int))
assert len(np.setdiff1d(fn_20x,idt_merge.query('is_new==True').oid))==0
idt_merge = idt_merge.sort_values(['is_new','is_anno'],ascending=False).reset_index(level=None, drop=True)
idt_merge.to_csv(os.path.join(dir_output,'idt_merge.csv'),index=False)

# 2x2 tables
print(idt_merge.groupby(['is_new','is_anno']).size().reset_index().rename(columns={0:'n'}))
print(idt_merge.groupby(['is_new','is_anno']).n_anno.sum().reset_index().query('n_anno>0'))

# Find the missing patients from the new batches
idt_merge.query('is_new==True & is_anno==False').drop(columns=['n_anno','is_anno','is_new']).idt.to_list()

# # Remaining rectal patients
# old_rectal = idt_merge.query('is_new==False & is_anno==False & tissue=="Rectum"')
# old_rectal = old_rectal[['idt','oid','tissue']].merge(df_nancy.drop(columns=['file','lab_dt']).rename(columns={'ID':'oid'}),'left',['oid','tissue'])
# old_rectal = old_rectal.drop(columns=['oid','score'])
# old_rectal[['CII','AIC','ULC']] = old_rectal[['CII','AIC','ULC']].astype(int)
# old_rectal


#############################
# ---- CIDSCAN BATCHES ---- #

from funs_support import zip_files

eosin_thresh = 10

df_cidscann = pd.read_excel(os.path.join(dir_ordinal,'cidscann_batches.xlsx'))
df_cidscann.insert(0,'idt',df_cidscann.file.str.split('\\s',1,True)[0])
fn_peak = pd.Series(os.listdir(dir_peak))
fn_peak = fn_peak[fn_peak.str.contains('^cleaned.*\\.png')].reset_index(None,True)
fn_peak = fn_peak.str.replace('.png','')
assert not fn_peak.duplicated().any()
idt_peak = pd.DataFrame({'fn':fn_peak,'idt':fn_peak.str.split('\\_',2,True)[1]})

# Loop through each and find
files_png = []
for i, idt in enumerate(df_cidscann.idt):
    # print('Iteration %i of %i' % (i+1, len(df_cidscann)))
    tmp_idt = idt_peak.query('idt == @idt').reset_index(None,True)
    assert len(tmp_idt) == 1
    path_csv = os.path.join(dir_peak, tmp_idt.loc[0,'fn'] + '.csv')
    path_png = tmp_idt.loc[0,'fn'] + '.png'
    tmp_inf = pd.read_csv(path_csv)
    eosin_max = tmp_inf.eosin.max()
    if eosin_max > 10:
        files_png.append(path_png)

zip_files(lst=files_png, fold=dir_peak, zip_fn='cidscann_v2.zip')

