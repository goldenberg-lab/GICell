import os
import numpy as np
import pandas as pd
from funs_support import find_dir_cell
import filecmp

dir_base = find_dir_cell()
dir_images = os.path.join(dir_base, 'images')
dir_output = os.path.join(dir_base, 'output')
dir_ordinal = os.path.join(dir_base,'..','GIOrdinal','data')
dir_cropped = os.path.join(dir_ordinal, 'cropped')

path_breaker = os.path.join(dir_ordinal,'df_codebreaker.csv')
df_breaker = pd.read_csv(path_breaker).drop(columns='file')
df_breaker['tissue2'] = df_breaker.file2.str.replace('.png','').str.split('_').apply(lambda x: x[-1])

# Image data on the GICell side
fn_images = pd.Series(os.listdir(dir_images))
fn_images = fn_images[fn_images.str.contains('png$')].reset_index(None,True)
df_images = fn_images.str.split('\\_',2,True).drop(columns=[0])
df_images.rename(columns={1:'idt',2:'tissue'}, inplace=True)
df_images['tissue'] = df_images.tissue.str.replace('.png','')
tmp = df_images.tissue.str.split('\\_|\\-',2,True)
tmp.rename(columns={0:'tissue',1:'num',2:'alt'}, inplace=True)
df_images = pd.concat([df_images.drop(columns='tissue'),tmp],1)
df_images['fn'] = fn_images.copy()

for ii, rr in df_images.iterrows():
    idt, tissue, fn = rr['idt'], rr['tissue'], rr['fn']
    path_GICell = os.path.join(dir_images, fn)
    match = df_breaker.query('QID==@idt & tissue==@tissue')
    tt = list(match['type'])[0]
    path_Ordinal = ''
    for tish in match.tissue2.unique():
        tmp_path = os.path.join(dir_cropped, tt, idt, tish)
        if os.path.exists(tmp_path):
            path_Ordinal = tmp_path
            break
    # Remove -v2
    fn2 = pd.Series(fn).str.replace('[\\-|\\_]v2','',regex=True)
    fn2 = fn2.str.replace(tissue,tish)[0]    
    path_Ordinal = os.path.join(path_Ordinal, fn2)
    assert os.path.exists(path_Ordinal)
    # Check that files are the same
    check = open(path_GICell,"rb").read() == open(path_Ordinal,"rb").read()
    if not check:
        print('file %s does not align' % fn)

    
    


