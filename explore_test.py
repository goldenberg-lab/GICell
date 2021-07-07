# Script to analyze performance of model on test set with both pixel-wise and clustered performance

import argparse
from plotnine.facets.facet_wrap import facet_wrap
from plotnine.positions.position_dodge import position_dodge

from plotnine.scales.scale_color import scale_color_datetime
from plotnine.themes.elements import element_text
parser = argparse.ArgumentParser()
parser.add_argument('--mdl_hash', type=str, help='How many points to pad around pixel annotation point')
args = parser.parse_args()
mdl_hash = args.save_model

# # For debugging
# mdl_hash = '4424974300780924119'
mdl_hash = None

# Remove .pkl if there
if mdl_hash is not None:
    mdl_hash = mdl_hash.split('.')[0]

import os
import numpy as np
import pandas as pd
import hickle
from time import time
from funs_support import find_dir_cell, hash_hp, makeifnot, read_pickle, no_diff, t2n, sigmoid
from cells import valid_cells, inflam_cells
from funs_stats import global_auroc
import plotnine as pn


dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')
lst_dir = [dir_output, dir_figures]
assert all([os.path.exists(z) for z in lst_dir])
dir_checkpoint = os.path.join(dir_output, 'checkpoint')

idx_eosin = np.where(pd.Series(valid_cells).isin(['eosinophil']))[0]
idx_inflam = np.where(pd.Series(valid_cells).isin(inflam_cells))[0]
di_idx = {'eosin':idx_eosin, 'inflam':idx_inflam}
di_cell = {'eosin':'Eosinophil', 'inflam':'Inflammatory'}

import torch
from funs_torch import img2tensor, all_img_flips
from funs_plotting import gg_save

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

pixel_max = 255        

img_trans = img2tensor(device)

dtype = np.float32

############################
## --- (1) LOAD MODEL --- ##

cells = ['inflam', 'eosin']
cn_hp = ['lr', 'p', 'batch']

# Find the "best" model is no mdl_hash provided
if mdl_hash is None:
    print('Finding "best" model')
    dat_ce = pd.read_csv(os.path.join(dir_output, 'dat_hp_ce.csv'))
    dat_ce = dat_ce.assign(value=lambda x: np.where(x.metric=='ce',-x.value,x.value))
    dat_auc = dat_ce.query('metric == "auc"')
    dat_auc_best = dat_auc.groupby(cn_hp).value.max().reset_index()
    dat_auc_hp = dat_auc_best.loc[[dat_auc_best.value.idxmax()]]
    fn_mdl = hash_hp(dat_auc_hp, method='hash_array')
    fn_mdl = str(fn_mdl) + '.pkl'
    # lr, p, batch = list(dat_auc_hp[cn_hp].itertuples(index=False,name=None))[0]
else:
    fn_mdl = mdl_hash + '.pkl'

# Load in mdls
di_fn = dict(zip(cells,[os.path.join(os.path.join(dir_checkpoint,cell),fn_mdl) for cell in cells]))
assert all([os.path.exists(v) for v in di_fn.values()])
di_mdl = {k1: {k2:v2 for k2, v2 in read_pickle(v1).items() if k2 in ['mdl','hp']} for k1, v1 in di_fn.items()}
dat_hp = pd.concat([v['hp'] for v in di_mdl.values()])
assert np.all(dat_hp.groupby('hp').val.var() == 0)
lr, p, batch = dat_hp.groupby('hp').val.mean()[cn_hp].to_list()
print('---- BEST HYPERPARAMETERS ----')
print('lr = %.3f, p = %i, batch = %i' % (lr, p, batch))
# Drop the hp and keep only model
di_mdl = {k: v['mdl'] for k, v in di_mdl.items()}
di_mdl = {k: v.eval() for k, v in di_mdl.items()}
di_mdl = {k: v.double() for k, v in di_mdl.items()}
# Models should be eval mode
assert all([not k.training for k in di_mdl.values()])

###########################
## --- (2) LOAD DATA --- ##

datasets = ['hsk', 'cinci']
# (i) Images + labels, di_data[ds][idt_tissue]
di_data = {ds: hickle.load(os.path.join(dir_output, 'annot_'+ds+'.pickle')) for ds in datasets}

# (ii) Aggregate cell counts
df_cells = pd.read_csv(os.path.join(dir_output, 'df_cells.csv'))

# Check alignment
assert df_cells.groupby('ds').apply(lambda x: no_diff(x.idt_tissue,list(di_data[x.ds.iloc[0]]))).all()

# Keep only eosin's + inflam
df_cells = df_cells[['ds','idt_tissue','eosinophil']].assign(inflam=df_cells[inflam_cells].sum(1)).rename(columns={'eosinophil':'eosin'})
print(df_cells.assign(sparse=lambda x: x.eosin == 0).groupby('ds').sparse.mean())

# (iii) Train/val/test
tmp1 = pd.read_csv(os.path.join(dir_output, 'train_val_test.csv'))
tmp1.insert(0, 'ds', 'hsk')
tmp2 = pd.DataFrame({'ds':'cinci','tt':'oos', 'idt_tissue':list(di_data['cinci'])})
tmp2 = df_cells.merge(tmp2,'right').assign(is_zero=lambda x: x.eosin == 0)
tmp2.drop(columns = cells, inplace=True)
df_tt = pd.concat([tmp1, tmp2], axis=0).reset_index(None, True)
print(df_tt.groupby('ds').is_zero.mean())

#######################################
## --- (3) INFERENCE STABILITY --- ##

stime = time()
holder = []
for ii , rr in df_tt.iterrows():
    ds, idt_tissue, tt = rr['ds'], rr['idt_tissue'], rr['tt']
    print('Row %i of %i' % (ii+1, len(df_tt)))
    # Load image/lbls
    img_ii = di_data[ds][idt_tissue]['img'].copy().astype(int)
    lbls_ii = di_data[ds][idt_tissue]['lbls'].copy().astype(float)
    assert img_ii.shape[:2] == lbls_ii.shape[:2]
    holder_cell = []
    # break
    for cell in cells:
        print('~~~~ cell = %s ~~~~' % cell)
        cell_ii = np.atleast_3d(lbls_ii[:,:,di_idx[cell]].sum(2))
        cell_ii_bin = np.squeeze(np.where(cell_ii > 0, 1, 0))
        # Get all rotations
        enc_all = all_img_flips(img_lbl=[img_ii, cell_ii], enc_tens=img_trans)
        enc_all.apply_flips()
        enc_all.enc2tensor()
        # To make sure GPU doesn't go over call in each batch
        holder_logits = torch.zeros(enc_all.lbl_tens.shape)
        for k in range(enc_all.ktot):
            tmp_img = enc_all.img_tens[[k],:,:,:] / 255
            with torch.no_grad():
                tmp_logits = di_mdl[cell](tmp_img)
            holder_logits[k, :, :, :] = tmp_logits
            torch.cuda.empty_cache()
        holder_sigmoid = sigmoid(t2n(holder_logits))
        holder_Y = np.where(t2n(enc_all.lbl_tens)>0,1,0)
        tmp_auc = [global_auroc(holder_Y[k,:,:,:], holder_sigmoid[k,:,:,:]) for k in range(enc_all.ktot)]
        tmp_df = enc_all.k_df.assign(auc=tmp_auc,cell=cell)
        # Reverse labels
        lst_img_lbl = img_trans.tensor2array([enc_all.img_tens, holder_logits])
        u_img, u_logits = enc_all.reverse_flips(img_lbl = lst_img_lbl)
        np.all(np.expand_dims(img_ii,3) == u_img)
        u_sigmoid = sigmoid(u_logits)
        mu_sigmoid = np.squeeze(np.apply_over_axes(np.mean, u_sigmoid, 3))
        mu_auc = global_auroc(cell_ii_bin, mu_sigmoid)
        tmp_df = tmp_df.append(pd.DataFrame({'rotate':-1,'flip':-1,'auc':mu_auc,'cell':cell},index=[0]))
        holder_cell.append(tmp_df)
    # Merge
    res_cell = pd.concat(holder_cell)
    res_cell = res_cell.assign(ds=ds, idt_tissue=idt_tissue, tt=tt)
    holder.append(res_cell)
    dtime = time() - stime
    nleft, rate = len(df_tt) - (ii+1), (ii+1) / dtime
    meta = (nleft / rate) / 60
    print('ETA: %.1f minutes, %.1f hours (%i of %i)' % (meta, meta/60, ii+1, len(df_tt)))
# Merge and save
inf_stab = pd.concat(holder).reset_index(None,True)
inf_stab.to_csv(os.path.join(dir_checkpoint, 'inf_stab.csv'),index=False)

# Plot it
tmp = inf_stab.assign(xlab=lambda x: 'r='+x.rotate.astype(str)+',f='+x.flip.astype(str))
tmp = tmp.assign(xlab=lambda x: np.where(x.rotate==-1,'mean',x.xlab))
tmp.drop(columns=['rotate','flip','ds'],inplace=True)

posd = pn.position_dodge(0.5)
gg_inf_stab = (pn.ggplot(tmp, pn.aes(x='xlab',y='auc',color='tt')) + 
    pn.theme_bw() + pn.geom_boxplot(position=posd) + 
    pn.facet_wrap('~cell',ncol=1,labeller=pn.labeller(cell=di_cell)) + 
    pn.labs(x='Rotate/Flip',y='AUROC',title='Pixel-wise') + 
    pn.scale_color_discrete(name='Type') + 
    pn.theme(axis_text_x=pn.element_text(angle=90)))
gg_save('gg_inf_stab.png', dir_figures, gg_inf_stab, 8, 8)


      
######################################
## --- (4) PIXEL-WISE INFERENCE --- ##

# (i) Plot figures
# (ii) Precision/recall



# Get the pixel sizes
h_pixel, w_pixel = di_data[df_cells.ds[0]][df_cells.idt_tissue[0]]['img'].shape[:2]
assert h_pixel == w_pixel

# array holder
arr_holder = np.zeros([len(df_tt), h_pixel, w_pixel])





# Holders for labels and phat
di_plbl = dict(zip(cells, [{'phat':arr_holder.copy(), 'lbls':arr_holder.copy()} for cell in cells]))

for ii , rr in df_tt.iterrows():
    print('Row %i of %i' % (ii+1, len(df_tt)))
    ds, idt_tissue, tt = rr['ds'], rr['idt_tissue'], rr['tt']
    # Load image/lbls
    img_ii = di_data[ds][idt_tissue]['img'].copy()
    lbls_ii = di_data[ds][idt_tissue]['lbls'].copy()
    assert img_ii.shape[:2] == lbls_ii.shape[:2]
    img_lbls_ii = [img_ii, lbls_ii]
    timg_ii, tlbls_ii = img_trans(img_lbls_ii)
    # UNet needs first dimension for batch-size
    timg_ii = torch.unsqueeze(timg_ii,dim=0) / pixel_max

    timg_ii.sum()


    # For each model get logits and labels
    for cell in cells:
        with torch.no_grad():
            tmp_phat = sigmoid(t2n(torch.squeeze(di_mdl[cell](timg_ii))))
            tmp_phat.flatten().shape
            timg_ii

        tmp_lbls = lbls_ii[:,:,di_idx[cell]].sum(2)
        assert tmp_phat.shape == tmp_lbls.shape
        # Store
        di_plbl[cell]['phat'][ii,:,:] = tmp_phat
        di_plbl[cell]['lbls'][ii,:,:] = tmp_lbls

# Save all predictions/labels in a hickle
path_plbl = os.path.join(dir_output, 'di_plbl.pickle')
hickle.dump(di_plbl, path_plbl, 'w')


########################################
## --- (4) PIXEL-WISE PERFORMANCE --- ##


    global_auroc(Ytrue,Ypred)
