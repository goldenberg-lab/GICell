# Find out which of the the possible images have received an annotation

import os
import pandas as pd
import numpy as np
from funs_support import find_dir_cell

dir_base = find_dir_cell()
dir_images = os.path.join(dir_base, 'images')
dir_points = os.path.join(dir_base, 'points')
dir_output = os.path.join(dir_base, 'output')
dir_ordinal = os.path.join(dir_base,'..','GIOrdinal','data')
dir_20x = os.path.join(dir_ordinal,'20x')
# Load in the code breaker
cn_breaker = ['ID','tissue','type','QID']
path_breaker = os.path.join(dir_ordinal,'df_codebreaker.csv')
df_breaker = pd.read_csv(path_breaker,usecols=cn_breaker)
df_breaker.rename(columns={'QID':'idt','type':'tt','ID':'oid'},inplace=True)
df_breaker = df_breaker.drop_duplicates().reset_index(None,True)

# Get the "new" images
fn_20x = pd.Series(os.listdir(dir_20x)).str.split('\\s',1,True)[0]
# Note that "SH19-1002" was a failed image so removing
fn_20x = pd.Series(np.setdiff1d(fn_20x,['SH19-1002']))

# Get the image list
fn_images = pd.Series(os.listdir(dir_images))
fn_images = fn_images[fn_images.str.contains('png$')].reset_index(None,True)

idt_images = fn_images.str.split('\\_',2,True).drop(columns=[0])
idt_images = pd.concat([idt_images[[1]], idt_images[2].str.replace('.png','').str.split('\\_',1,True)[[0]]],1)
idt_images.rename(columns={1:'idt',0:'tissue'},inplace=True)
idt_images = idt_images.drop_duplicates().reset_index(None,True)
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
idt_merge = idt_merge.sort_values(['is_new','is_anno'],ascending=False).reset_index(None,True)
idt_merge.to_csv(os.path.join(dir_output,'idt_merge.csv'),index=False)

# 2x2 tables
idt_merge.groupby(['is_new','is_anno']).size().reset_index().rename(columns={0:'n'})
idt_merge.groupby(['is_new','is_anno']).n_anno.sum().reset_index().query('n_anno>0')

