# Script to analyze performance of model on test set with both pixel-wise and clustered performance

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--mdl_hash', type=str, help='How many points to pad around pixel annotation point')
parser.add_argument('--nfill', type=int, default=1, help='How many points to pad around pixel annotation point')
parser.add_argument('--check_flips', dest='check_int', action='store_true', help='Compare inference for different angles/rotations')
args = parser.parse_args()
mdl_hash = args.save_model
check_flips = args.check_flips

# # For debugging
# mdl_hash = None
# mdl_hash = '4424974300780924119'
nfill = 1
check_flips = False

# # Remove .pkl if there
# if mdl_hash is not None:
#     mdl_hash = mdl_hash.split('.')[0]

import os
import numpy as np
import pandas as pd
import hickle
from time import time
from funs_support import find_dir_cell, makeifnot, read_pickle, no_diff, t2n, sigmoid, makeifnot
from cells import valid_cells, inflam_cells
from funs_stats import global_auroc, global_auprc, rho, get_num_label, lbl_freq
import plotnine as pn
from scipy import stats
from sklearn.metrics import r2_score

dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')
dir_inference = os.path.join(dir_figures, 'inference')
makeifnot(dir_inference)

lst_dir = [dir_output, dir_figures]
assert all([os.path.exists(z) for z in lst_dir])
dir_best = os.path.join(dir_output, 'best')
dir_checkpoint = os.path.join(dir_output, 'checkpoint')

idx_eosin = np.where(pd.Series(valid_cells).isin(['eosinophil']))[0]
idx_inflam = np.where(pd.Series(valid_cells).isin(inflam_cells))[0]
di_idx = {'eosin':idx_eosin, 'inflam':idx_inflam}
di_cell = {'eosin':'Eosinophil', 'inflam':'Inflammatory'}

import torch
from funs_torch import img2tensor, all_img_flips
from funs_plotting import gg_save, post_plot

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

pixel_max = 255        
img_trans = img2tensor(device)
dtype = np.float32

# number of padded points (i.e. count inflator)
fillfac = (2 * nfill + 1) ** 2
print('nfill: %i, fillfac: x%i' % (nfill, fillfac))


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
# Models should be eval mode
assert all([not k.training for k in di_mdl.values()])

di_tt = {'train':'Train','val':'Val','test':'Test','oos':'Cinci'}

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
df_cells_long = df_cells.melt(['ds','idt_tissue'],None,'cell','act')

# (iii) Train/val/test
tmp1 = pd.read_csv(os.path.join(dir_output, 'train_val_test.csv'))
tmp1.insert(0, 'ds', 'hsk')
tmp2 = pd.DataFrame({'ds':'cinci','tt':'oos', 'idt_tissue':list(di_data['cinci'])})
tmp2 = df_cells.merge(tmp2,'right').assign(is_zero=lambda x: x.eosin == 0)
tmp2.drop(columns = cells, inplace=True)
df_tt = pd.concat([tmp1, tmp2], axis=0).reset_index(None, True)
print(df_tt.groupby('ds').is_zero.mean())
df_tt_idt = df_tt[['idt_tissue','tt']].copy()

#######################################
## --- (3) INFERENCE STABILITY --- ##

if check_flips:
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
            enc_all = all_img_flips(img_lbl=[img_ii, cell_ii], enc_tens=img_trans, tol=0.1)
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
    inf_stab.tt = pd.Categorical(inf_stab.tt,['train','val','test','oos']).map(di_tt)

    inf_stab.assign(is_na=lambda x: x.auc.isnull()).groupby(['tt','cell']).is_na.mean()

    # Plot it
    tmp = inf_stab.assign(xlab=lambda x: 'r='+x.rotate.astype(str)+',f='+x.flip.astype(str))
    tmp = tmp.assign(xlab=lambda x: np.where(x.rotate==-1,'mean',x.xlab))
    tmp.drop(columns=['rotate','flip','ds'],inplace=True)

    posd = pn.position_dodge(0.75)
    gg_inf_stab = (pn.ggplot(tmp, pn.aes(x='xlab',y='auc',color='tt')) + 
        pn.theme_bw() + pn.geom_boxplot(position=posd) + 
        pn.facet_wrap('~cell',ncol=1,labeller=pn.labeller(cell=di_cell)) + 
        pn.labs(x='Rotate/Flip',y='AUROC',title='Pixel-wise') + 
        pn.scale_color_discrete(name='Type',labels=['Train','Val','Test','Cinci']) + 
        pn.theme(axis_text_x=pn.element_text(angle=90)))
    gg_save('gg_inf_stab.png', dir_figures, gg_inf_stab, 12, 8)

    # Determine and AUROC gain
    cn_gg = ['tt','cell','is_mu']
    tmp = inf_stab.assign(is_mu=lambda x: x.rotate<0).groupby(cn_gg)
    dat_ttest = tmp.auc.apply(lambda x: pd.Series({'auc':x.mean(), 'n':len(x),'se':x.std(ddof=1)}))
    dat_ttest = dat_ttest.reset_index().pivot_table('auc',cn_gg,'level_3').reset_index()
    dat_ttest.is_mu = dat_ttest.is_mu.astype(str)
    dat_ttest = dat_ttest.pivot_table(['auc','n','se'],['tt','cell'],'is_mu').reset_index()
    cn = pd.Series(['_'.join(col).strip() for col in dat_ttest.columns.values]).str.replace('\\_$','',regex=True)
    dat_ttest.columns = cn
    dat_ttest = dat_ttest.assign(d_auc=lambda x: x.auc_True - x.auc_False,
        se_d=lambda x: np.sqrt(x.se_True**2/x.n_True + x.se_False**2/x.n_False) )
    dat_ttest.d_auc.mean()

    # Standard AUROC
    critv = stats.norm.ppf(0.975)
    res_auc = inf_stab.query('rotate==0 & flip==0').drop(columns=['rotate','flip']).dropna().groupby(['tt','cell'])
    res_auc = res_auc.auc.apply(lambda x: pd.Series({'auc':x.mean(), 'n':len(x),'se':x.std(ddof=1)}))
    res_auc = res_auc.reset_index().pivot_table('auc',['tt','cell'],'level_2').reset_index()
    res_auc = res_auc.assign(lb=lambda x: x.auc-critv*x.se/np.sqrt(x.n), ub=lambda x: x.auc+critv*x.se/np.sqrt(x.n))

    position = pn.position_dodge(0.5)
    gg_auroc_tt = (pn.ggplot(res_auc, pn.aes(x='tt',y='auc',color='cell')) + 
    pn.theme_bw() + pn.labs(y='Average AUROC',title='Pixel-wise') + 
    pn.geom_point(position=posd,size=2) + 
    pn.theme(axis_title_x=pn.element_blank()) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
    pn.scale_y_continuous(limits=[0.8,1],breaks=list(np.arange(0.8,1.01,0.05))) + 
    pn.scale_color_discrete(name='Type',labels=['Eosinophil','Inflammatory']))
    gg_save('gg_auroc_tt.png', dir_figures, gg_auroc_tt, 6, 4)

#############################################
## --- (4) PIXEL-WISE PRECISION/RECALL --- ##

# Loop through and save all pixel-wise probabilities
h, w, c = di_data[df_tt.ds[0]][df_tt.idt_tissue[0]]['img'].shape
assert h == w

# Numpy array to store results: (h x w x # cells x # patients)
holder_logits = np.zeros([len(df_tt), len(cells), h, w])
holder_lbls = holder_logits.copy()

stime = time()
for ii , rr in df_tt.iterrows():
    ds, idt_tissue, tt = rr['ds'], rr['idt_tissue'], rr['tt']
    print('Row %i of %i' % (ii+1, len(df_tt)))
    # Load image/lbls
    img_ii = di_data[ds][idt_tissue]['img'].copy()
    lbls_ii = di_data[ds][idt_tissue]['lbls'].copy()
    assert img_ii.shape[:2] == lbls_ii.shape[:2]
    timg_ii, tlbls_ii = img_trans([img_ii, lbls_ii])
    timg_ii = torch.unsqueeze(timg_ii,0) / 255
    for jj, cell in enumerate(cells):
        cell_ii = np.atleast_3d(lbls_ii[:,:,di_idx[cell]].sum(2))
        cell_ii_bin = np.squeeze(np.where(cell_ii > 0, 1, 0))
        ncell_ii = cell_ii.sum() / fillfac
        #print('~~~~ cell = %s, n=%i ~~~~' % (cell, ncell_ii))
        with torch.no_grad():
            tmp_logits = t2n(di_mdl[cell](timg_ii))
        # Save for later
        holder_logits[ii,jj,:,:] = tmp_logits
        holder_lbls[ii,jj,:,:] = cell_ii_bin
    torch.cuda.empty_cache()
    dtime = time() - stime
    nleft, rate = len(df_tt) - (ii+1), (ii+1) / dtime
    meta = (nleft / rate) / 60
    print('ETA: %.1f minutes (%i of %i)' % (meta, ii+1, len(df_tt)))

# Calculate precision/recall for each cell type
holder = []
for jj, cell in enumerate(cells):
    for tt in di_tt:
        print('~~~~ cell = %s, tt=%s ~~~~' % (cell, tt))
        idx_tt = df_tt.query('tt==@tt').idt_tissue.index.values
        phat_tt = sigmoid(holder_logits[idx_tt, jj, :, :])
        ybin_tt = holder_lbls[idx_tt, jj, :, :]
        tmp_df = global_auprc(ybin_tt, phat_tt, n_points=50)
        tmp_df = tmp_df.assign(cell=cell, tt=tt)
        holder.append(tmp_df)
res_pr = pd.concat(holder).melt(['cell','tt','thresh'],None,'msr')
res_pr.tt = pd.Categorical(res_pr.tt, list(di_tt)).map(di_tt)
res_pr.cell = res_pr.cell.map(di_cell)

# Make a plot
gg_auprc = (pn.ggplot(res_pr, pn.aes(x='thresh',y='value',color='tt',linetype='msr')) + 
    pn.theme_bw() + pn.geom_line() + 
    pn.facet_wrap('~cell',nrow=1,scales='free_x') + 
    pn.scale_color_discrete(name='Dataset') + 
    pn.scale_linetype_manual(name='Measure',labels=['Precision','Recall'],values=['solid','dashed']) + 
    pn.labs(y='Percent',x='Threshold'))
gg_save('gg_auprc.png', dir_figures, gg_auprc, 10, 4)


########################################
## --- (5) PROBABILITY CLUSTERING --- ##

conn_seq = [1, 2]

nquant = 20
idt_train = df_tt.query('tt=="train"').idt_tissue
idt_rest = df_tt.query('tt!="train"').idt_tissue
idx_train = idt_train.index.values
idx_rest = idt_rest.index.values
# Loop over
holder = []
for jj, cell in enumerate(cells):
    phat_train = sigmoid(holder_logits[idx_train, jj, :, :])
    phat_rest = sigmoid(holder_logits[idx_rest, jj, :, :])
    ntrain, nrest = len(phat_train), len(phat_rest)
    ybin_train = holder_lbls[idx_train, jj, :, :]
    thresh_train = np.quantile(phat_train[ybin_train==1], np.linspace(0, 1, nquant)[1:-1])
    for kk, thresh in enumerate(thresh_train):
        print('~~~~ cell = %s, thresh %i of %i ~~~~' % (cell, kk+1, nquant))
        yhat_train = np.where(phat_train >= thresh, 1, 0)
        yhat_rest = np.where(phat_rest >= thresh, 1, 0)
        for conn in conn_seq:
            tmp_train = pd.concat([lbl_freq(yhat_train[i], conn, idt) for i, idt in zip(range(ntrain),idt_train)])
            tmp_rest = pd.concat([lbl_freq(yhat_rest[i], conn, idt) for i, idt in zip(range(nrest),idt_rest)])
            # Merge and annotate
            tmp_both = pd.concat([tmp_train, tmp_rest], axis=0).assign(cell=cell, thresh=thresh, conn=conn)
            holder.append(tmp_both)
# Combine all results
res_cell = pd.concat(holder).reset_index(None, True)
# Merge with dataset type
res_cell.rename(columns={'idx':'idt_tissue'}, inplace=True)
res_cell = res_cell.merge(df_tt_idt)
# res_cell = res_cell.merge(df_cells_long.drop(columns='ds'))


# Loop over each combination and the relationship between the n cut-off
q_seq = np.arange(0.05,1,0.05)
holder = []
for jj, cell in enumerate(cells):
    tmp_df1 = res_cell.query('cell==@cell')
    thresh_train = tmp_df1.thresh.unique()
    for kk, thresh in enumerate(thresh_train):
        for conn in conn_seq:
            print('cell=%s, thresh=%.5f, conn=%i' % (cell, thresh, conn))
            tmp_df2 = tmp_df1.query('conn==@conn & thresh==@thresh').reset_index(None, True)
            n_seq = np.sort(tmp_df2.query('tt=="val"').n.unique())
            if len(n_seq) > len(q_seq):
                n_seq = np.quantile(n_seq, q_seq)
            for n in n_seq:
                tmp_n = tmp_df2.groupby('idt_tissue').apply(lambda x: np.sum(x.n >= n)).reset_index()
                tmp_n = tmp_n.rename(columns={0:'est'}).assign(cell=cell)
                tmp_n = tmp_n.merge(df_cells_long).merge(df_tt_idt)
                tmp_n = tmp_n.groupby('tt').apply(lambda x: pd.Series({'rho':rho(x.act, x.est),'r2':r2_score(x.act, x.est)}))
                tmp_n = tmp_n.reset_index().assign(cell=cell,thresh=kk,conn=conn,n=n)
                holder.append(tmp_n)
# Merge and visually inspect trade-off
res_rho = pd.concat(holder).reset_index(None,True)
res_rho.tt = pd.Categorical(res_rho.tt,list(di_tt)).map(di_tt)
# Melt on rho vs r2
res_rho = res_rho.melt(['tt','cell','thresh','conn','n'],None,'msr')
# Clip the r-squared
res_rho.value = res_rho.value.clip(-1,1)

##############################
## --- (6) FIND OPTIMAL --- ##

# --- (i) Find the "best" point for each cell/dataset --- #
res_rho_star = res_rho.groupby(['cell','msr','thresh','conn','n']).value.mean().reset_index()
res_rho_star = res_rho_star.loc[res_rho_star.groupby(['cell','msr']).apply(lambda x: x.value.idxmax())]
res_rho_star = res_rho.merge(res_rho_star.drop(columns='value'),'inner')

holder_n = []
di_conn = {'logits':holder_logits.copy(), 'post':np.zeros(holder_logits.shape),
           'idt_tissue':df_tt.idt_tissue.values,
           'thresh':np.zeros(2), 'conn':np.zeros(2), 'n':np.zeros(2)}
for jj, cell in enumerate(cells):
    phat_train = sigmoid(holder_logits[idx_train, jj, :, :])
    phat_rest = sigmoid(holder_logits[idx_rest, jj, :, :])
    tmp_star = res_rho_star.query('cell == @cell')
    ybin_train = holder_lbls[idx_train, jj, :, :]
    thresh_train = np.quantile(phat_train[ybin_train==1], np.linspace(0, 1, nquant)[1:-1])
    for msr in tmp_star.msr.unique():
        print('~~~~ cell = %s, msr=%s ~~~~' % (cell, msr))
        tmp_star_msr = tmp_star.query('msr==@msr').reset_index(None, True)
        thresh_idx_star = tmp_star_msr.thresh[0]
        n_star = tmp_star_msr.n[0]
        conn_star = tmp_star_msr.conn[0]
        thresh_star = thresh_train[thresh_idx_star]
        # Apply threshold
        yhat_train = np.where(phat_train >= thresh_star, 1, 0)
        yhat_rest = np.where(phat_rest >= thresh_star, 1, 0)
        # Connect connectivity map
        tmp_train = pd.concat([lbl_freq(yhat_train[i], conn_star, idt) for i, idt in zip(range(ntrain),idt_train)])
        tmp_rest = pd.concat([lbl_freq(yhat_rest[i], conn_star, idt) for i, idt in zip(range(nrest),idt_rest)])
        tmp_both = pd.concat([tmp_train, tmp_rest], axis=0)
        tmp_both = tmp_both.groupby('idx').apply(lambda x: np.sum(x.n >= n_star))
        tmp_both = tmp_both.reset_index().rename(columns={'idx':'idt_tissue',0:'est'})
        tmp_both = tmp_both.assign(cell=cell).merge(df_cells_long).merge(df_tt_idt)
        tmp_both = tmp_both.assign(thresh=thresh_star, n=n_star, conn=conn_star, msr=msr)
        holder_n.append(tmp_both)
        # Use the R-squared model
        if msr == 'r2':
            di_conn['thresh'][jj] = thresh_star
            di_conn['conn'][jj] = conn_star
            di_conn['n'][jj] = n_star
            tmp1 = np.stack([phat2lbl(yhat_train[i], thresh_star, n_star, conn_star) for i in range(ntrain)],axis=0)
            tmp2 = np.stack([phat2lbl(yhat_rest[i], thresh_star, n_star, conn_star) for i in range(nrest)],axis=0)
            di_conn['post'][:, jj, :, :] = np.vstack([tmp1, tmp2])
            del tmp1, tmp2
res_scatter = pd.concat(holder_n).reset_index(None, True)
res_scatter.tt = pd.Categorical(res_scatter.tt,list(di_tt)).map(di_tt)

# Save for later
path_conn = os.path.join(dir_output,'di_conn.pickle')
hickle.dump(di_conn, path_conn, 'w')

# di_conn = hickle.load(path_conn)
# {k:v.shape for k,v in di_conn.items()}

# https://scikit-image.org/docs/dev/auto_examples/applications/plot_human_mitosis.html#sphx-glr-auto-examples-applications-plot-human-mitosis-py

######################
## --- (7) UNET --- ##

df_unet = np.squeeze(np.apply_over_axes(np.sum, sigmoid(di_conn['logits']), [2,3]))
df_unet = pd.DataFrame(df_unet / fillfac, columns=cells)
df_unet.insert(0,'idt_tissue', di_conn['idt_tissue'])
df_unet = df_unet.melt('idt_tissue',None,'cell','est').merge(df_cells_long).merge(df_tt_idt)
df_unet['cell'] = df_unet['cell'].map(di_cell)
df_unet['tt'] = pd.Categorical(df_unet['tt'],list(di_tt)).map(di_tt)

df_unet_rho = df_unet.groupby(['cell','tt']).apply(lambda x: pd.Series({'r2':r2_score(x.act, x.est), 'rho':rho(x.act, x.est)}))
df_unet_rho.reset_index(inplace=True)
df_unet_rho['txt'] = df_unet_rho.apply(lambda x: 'R2=%.1f%%, Rho=%.1f%%' % (x.r2*100,x.rho*100),1)


#########################
## --- (8) FIGURES --- ##

# --- (v) UNet scatter --- #
gg_scatter_unet = (pn.ggplot(df_unet,pn.aes(x='est',y='act',color='tt')) + 
    pn.theme_bw() + pn.geom_point() + 
    pn.scale_color_discrete(name='Dataset') + 
    pn.geom_abline(slope=1,intercept=0,color='black',linetype='--') + 
    pn.facet_wrap('~cell+tt',scales='free',nrow=2) + 
    pn.theme(subplots_adjust={'wspace': 0.20, 'hspace':0.20}) + 
    pn.guides(color=False) + 
    pn.labs(x='Predicted',y='Actual'))
gg_save('gg_scatter_unet.png', dir_figures, gg_scatter_unet, 13, 6)


# --- (iv) Actual, phat, post-proc --- #
idt_test = df_tt_idt[df_tt_idt.tt.isin(['test','oos'])].idt_tissue.to_list()
for kk, idt in enumerate(idt_test):
    print('image %i of %i (%s)' % (kk+1,len(idt_test),idt))
    # Find indexes
    ii = np.where(di_conn['idt_tissue'] == idt)[0][0]
    ds, tt = df_tt.query('idt_tissue==@idt')[['ds','tt']].values.flat
    # Load images/labels
    img_ii = di_data[ds][idt]['img']
    lbls_ii = np.dstack([di_data[ds][idt]['lbls'][:,:,di_idx[cell]].sum(2) for cell in cells])
    # Extract phat/yhat
    logits_ii = di_conn['logits'][ii].transpose(1,2,0)    
    post_ii = di_conn['post'][ii].transpose(1,2,0)
    phat_ii = sigmoid(logits_ii)
    yhat_ii = np.where(phat_ii >= np.expand_dims(di_conn['thresh'],[0,1]),1,0)
    # Plot it
    fn_idt = ds + '_' + tt + '_' + idt + '.png'
    post_plot(img=img_ii, lbls=lbls_ii, phat=phat_ii, yhat=yhat_ii, 
              fillfac=fillfac, cells=cells, thresh=di_conn['thresh'],
              fold=dir_inference, fn=fn_idt, title=idt)


# --- (i) Performance as a function of n and thresh --- #
for msr in ['r2','rho']:
    tmp_df = res_rho.query('msr == @msr').dropna()
    tmp_star = res_rho_star.query('msr == @msr')
    tmp_fn = 'gg_thresh_n_' + msr + '.png'
    tmp_gg = (pn.ggplot(tmp_df, pn.aes(x='n',y='value',color='tt')) + 
        pn.theme_bw() + pn.geom_line() + 
        pn.scale_y_continuous(limits=[-1,1]) + 
        pn.facet_wrap('~cell + thresh',scales='free_x') + 
        pn.scale_color_discrete(name='Dataset') + 
        # pn.scale_shape_discrete(name='Connectivity') + 
        pn.geom_hline(yintercept=0,linetype='--') + 
        pn.geom_hline(yintercept=0.5,linetype='--',color='darkgreen') + 
        pn.ggtitle('Measure = %s' % msr) + 
        pn.geom_point(pn.aes(x='n',y='value',color='tt'),data=tmp_star,inherit_aes=False,size=2) + 
        pn.theme(subplots_adjust={'hspace': 0.5, 'wspace': 0.1}) +
        pn.labs(y=msr,x='Minimum cell count'))
    gg_save(tmp_fn, dir_figures, tmp_gg, 15, 15)


# --- (ii) Visualize the "best" scatter --- #
gg_scatter_star = (pn.ggplot(res_scatter,pn.aes(x='est',y='act',color='tt')) + 
    pn.theme_bw() + pn.geom_point() + 
    pn.scale_color_discrete(name='Dataset') + 
    pn.geom_abline(slope=1,intercept=0,color='black',linetype='--') + 
    pn.facet_wrap('~cell+msr+tt',scales='free', nrow=4) + 
    pn.theme(subplots_adjust={'wspace': 0.20, 'hspace':0.50}) + 
    pn.guides(color=False) + 
    pn.labs(x='Predicted',y='Actual'))
gg_save('gg_scatter_star.png', dir_figures, gg_scatter_star, 13, 12)


# --- (iii) Point estimate and CI for correlation/R-squared --- #
n_bs = 250
alpha = 0.05
cn_gg = ['cell','tt','msr']
holder_bs = []
for i in range(n_bs):
    for msr in res_scatter.msr.unique():
        tmp_msr = res_scatter.query('msr == @msr')
        tmp_df = tmp_msr.groupby(cn_gg).sample(frac=1,replace=True, random_state=i)
        tmp_df = tmp_df.groupby(cn_gg).apply(lambda x: 
            pd.Series({'r2':r2_score(x.act, x.est), 'rho':rho(x.act, x.est)}))
        tmp_df = tmp_df[msr].reset_index().rename(columns={msr:'value'}).assign(bidx=i)
        holder_bs.append(tmp_df)
# Merge
tmp_df = pd.concat(holder_bs).groupby(cn_gg).value.quantile([alpha/2, 1-alpha/2])
tmp_df = tmp_df.reset_index().pivot_table('value',cn_gg,'level_3')
tmp_df = tmp_df.reset_index().rename(columns={alpha/2:'lb', 1-alpha/2:'ub'})
res_cell_perf = res_rho_star.merge(tmp_df)

# Show the metrics
posd = pn.position_dodge(0.5)
gg_perf_star = (pn.ggplot(res_cell_perf,pn.aes(x='msr',y='value',color='cell')) + 
    pn.theme_bw() + pn.geom_point(position=posd) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
    pn.scale_color_discrete(name='Cell') + 
    pn.labs(y='Percent',x='Metric') + 
    pn.geom_hline(yintercept=0,linetype='--') + 
    pn.ggtitle('Linerange shows 95% CI') + 
    pn.theme(subplots_adjust={'wspace': 0.25}) + 
    pn.facet_wrap('~tt',nrow=2))
gg_save('gg_perf_star.png', dir_figures, gg_perf_star, 6, 5)
# ,scales='free_y'





