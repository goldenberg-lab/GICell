# Script to analyze performance of model on test set with both pixel-wise and clustered performance

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nfill', type=int, default=1, help='How many points to pad around pixel annotation point')
args = parser.parse_args()
nfill = args.nfill
check_flips = args.check_flips
ds_test = args.ds_test
print('args : %s' % args)

# # For debugging
# nfill = 1

import os
import numpy as np
import pandas as pd
import hickle
from time import time
from funs_support import find_dir_cell, makeifnot, read_pickle, t2n, sigmoid, makeifnot
from cells import valid_cells, inflam_cells, di_ds, di_tt
from funs_stats import global_auroc, global_auprc, rho, phat2lbl, lbl_freq
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

# Load in the "best" models for each cell type
cells = os.listdir(dir_checkpoint)
n_cells = len(cells)

di_mdl = dict.fromkeys(cells)
for cell in cells:
    dir_cell = os.path.join(dir_checkpoint, cell)
    fn_cell = pd.Series(os.listdir(dir_cell))
    fn_cell = fn_cell[fn_cell.str.contains('best_%s.pkl'%cell)]
    assert len(fn_cell) == 1, 'Could not detect single best model'
    fn_best = fn_cell.to_list()[0]
    path_best = os.path.join(dir_cell, fn_best)
    assert os.path.exists(path_best)
    di_mdl[cell] = read_pickle(path_best)
# Extract the hyperparameters
dat_hp = [v['hp'].assign(cell=k) for k, v in di_mdl.items()]
dat_hp = pd.concat(dat_hp).reset_index(None,drop=True)
print('---- BEST HYPERPARAMETERS ----')
print(dat_hp)
# Drop the hp and keep only model
di_mdl = {k: v['mdl'] for k, v in di_mdl.items()}
# If model is wrapped in DataParallel extract model
di_mdl = {k: v.module if hasattr(v,'module') else v for k, v in di_mdl.items() }
di_mdl = {k: v.eval() for k, v in di_mdl.items()}
di_mdl = {k: v.float() for k, v in di_mdl.items()}
# Models should be eval mode
assert all([not k.training for k in di_mdl.values()])


###########################
## --- (2) LOAD DATA --- ##

# (i) Aggregate cell counts
df_cells = pd.read_csv(os.path.join(dir_output, 'df_cells.csv'))
u_ds = list(df_cells['ds'].unique())
# Keep only eosin's + inflam
df_cells = df_cells.assign(inflam=df_cells[inflam_cells].sum(1))
df_cells.rename(columns={'eosinophil':'eosin'}, inplace=True)
df_cells.drop(columns=valid_cells, inplace=True, errors='ignore')
df_cells_long = df_cells.melt(['ds','idt'],['eosin','inflam'],'cell','act')
pct_sparse = df_cells_long.groupby(['cell','ds']).apply(lambda x: pd.Series({'sparse':np.mean(x.act == 0)}))
pct_sparse = pct_sparse.sort_values(['cell','sparse']).reset_index()
print(pct_sparse.reset_index().round(3))

# (ii) Train/val/test
df_sets = pd.read_csv(os.path.join(dir_output, 'train_val_test.csv'),usecols=['ds','idt','tt'])
n_sets = len(df_sets)
u_tt = list(df_sets['tt'].unique())
df_ds_tt_idt = df_sets[['tt','ds','idt']].copy()

# (iii) Load the test data
di_data = dict.fromkeys(u_ds)
for ds in u_ds:
    print('ds = %s' % ds)
    path_ds = os.path.join(dir_output, 'annot_%s.pickle'%ds)
    di_data[ds] = hickle.load(path_ds)

# Check that df_cells and di_data lines up
idt_di_data = pd.concat([pd.DataFrame({'ds':k,'idt':list(v.keys())}) for k,v in di_data.items()]).reset_index(None,drop=True)
check1 = idt_di_data.assign(val1=1).merge(df_cells[['idt','ds']].assign(val2=1),'left')
assert check1['val2'].notnull().all(), 'Not all idts line up'

# (iv) Re-order data dictionary in terms of train/val/test
di_tt_ds = dict.fromkeys(u_tt)
for tt in u_tt:
    tmp_ds = df_sets[df_sets['tt'] == tt]['ds'].unique()
    di_tt_ds[tt] = dict.fromkeys(tmp_ds)
    for ds in tmp_ds:
        print('tt=%s, ds=%s' % (tt, ds))
        tmp_sets = df_sets.query('tt==@tt & ds==@ds')
        tmp_idt = tmp_sets['idt']
        assert not tmp_idt.duplicated().any(), 'Duplicated idts'
        di_tt_ds[tt][ds] = dict.fromkeys(tmp_idt)
        for idt in tmp_idt:
            di_tt_ds[tt][ds][idt] = di_data[ds][idt]


#######################################
## --- (3) PIXELWISE PERFORMANCE --- ##

tol_pct, tol_dcell = 0.02, 2
jj = 0
stime = time()
holder = []
for tt in di_tt_ds:
    for ds in di_tt_ds[tt]:
        print('--- tt=%s, ds=%s ---' % (tt, ds))
        tmp_idt = list(di_tt_ds[tt][ds])
        n_idt = len(tmp_idt)
        for ii, idt in enumerate(tmp_idt):
            print('Patient %i of %i' % (ii+1, n_idt))
            jj += 1
            # Extract labels and images
            img, lbls = di_tt_ds[tt][ds][idt].values()
            tmp_cells = df_cells.query('ds==@ds & idt==@idt')
            if np.any(tmp_cells[cells] != 0):
                # Cell-wise inference
                holder_cell = []
                for cell in cells:
                    lbl_cell = np.atleast_3d(lbls[:,:,di_idx[cell]].sum(2))
                    lbl_cell_bin = np.squeeze(np.where(lbl_cell > 0, 1, 0))
                    est_lbl = int(np.round(lbl_cell.sum() / fillfac))
                    est_df = tmp_cells[cell].values[0]
                    if (est_lbl == 0) & (est_df == 0):
                        print('Skipping AUC calculation, zero cells') 
                    else:
                        check1 = np.abs(est_lbl - est_df) <= tol_dcell
                        check2 = np.abs(est_lbl / est_df - 1) < tol_pct
                        assert check1 or check2, 'Cell discrepancy detected'

                        # Get rotations
                        enc_all = all_img_flips(img_lbl=[img, lbl_cell], enc_tens=img_trans, tol=0.1)
                        enc_all.apply_flips()
                        enc_all.enc2tensor()
                        
                        # Do inference
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
                        # Reverse labels back to original orientation
                        lst_img_lbl = img_trans.tensor2array([enc_all.img_tens, holder_logits])
                        u_img, u_logits = enc_all.reverse_flips(img_lbl = lst_img_lbl)
                        assert np.all(np.expand_dims(img,3) == u_img), 'Reverse image failed'
                        u_sigmoid = sigmoid(u_logits)
                        mu_sigmoid = np.squeeze(np.apply_over_axes(np.mean, u_sigmoid, 3))
                        mu_auc = global_auroc(lbl_cell_bin, mu_sigmoid)
                        # Represents averaging of all flips/rotations
                        tmp_auc_mu = pd.DataFrame({'rotate':-1,'flip':-1,'auc':mu_auc,'cell':cell},index=[0])
                        tmp_df = tmp_df.append(tmp_auc_mu)
                        holder_cell.append(tmp_df)
                res_cell = pd.concat(holder_cell)
                res_cell = res_cell.assign(tt=tt, ds=ds, idt=idt)
                holder.append(res_cell)
                dtime = time() - stime
                nleft, rate = n_sets - jj, jj / dtime
                meta = (nleft / rate) / 60
                print('ETA: %.1f minutes (%i of %i)' % (meta, jj, n_sets))

# Merge and save
inf_stab = pd.concat(holder).reset_index(None,drop=True)
inf_stab.to_csv(os.path.join(dir_output, 'inf_stab.csv'),index=False)
# Convert ds/tt to labels
inf_stab['ds'] = inf_stab['ds'].map(di_ds)
inf_stab['tt'] = pd.Categorical(inf_stab['tt'],list(di_tt)).map(di_tt)

# Plot distribution of AUCs
tmp = inf_stab.assign(xlab=lambda x: 'r='+x.rotate.astype(str)+',f='+x.flip.astype(str))
tmp = tmp.assign(xlab=lambda x: np.where(x.rotate==-1,'mean',x.xlab))
tmp.drop(columns=['rotate','flip','ds'],inplace=True)

posd = pn.position_dodge(0.75)
gg_inf_stab = (pn.ggplot(tmp, pn.aes(x='xlab',y='auc',color='tt')) + 
    pn.theme_bw() + pn.geom_boxplot(position=posd) + 
    pn.facet_wrap('~cell',ncol=1,labeller=pn.labeller(cell=di_cell)) + 
    pn.labs(x='Rotate/Flip',y='AUROC',title='Pixel-wise') + 
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
print('AUC gain of averaging rotations: %.3f' % dat_ttest.d_auc.mean())

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
h_max = df_cells['h'].max()
w_max = df_cells['w'].max()

# Numpy array to store results: (h x w x # cells x # patients)
holder_logits = np.zeros([n_sets, len(cells), h_max, w_max]) * np.nan
holder_lbls = holder_logits.copy()

stime = time()
for ii , rr in df_sets.iterrows():
    ds, idt, tt = rr['ds'], rr['idt'], rr['tt']
    # print('Row %i of %i' % (ii+1, n_sets))
    # Load image/lbls
    img_ii = di_tt_ds[tt][ds][idt]['img'].copy()
    lbls_ii = di_tt_ds[tt][ds][idt]['lbls'].copy()
    assert img_ii.shape[:2] == lbls_ii.shape[:2]
    timg_ii, tlbls_ii = img_trans([img_ii, lbls_ii])
    timg_ii = torch.unsqueeze(timg_ii,0) / 255
    for jj, cell in enumerate(cells):
        cell_ii = np.atleast_3d(lbls_ii[:,:,di_idx[cell]].sum(2))
        cell_ii_bin = np.squeeze(np.where(cell_ii > 0, 1, 0))
        ncell_ii = cell_ii.sum() / fillfac
        with torch.no_grad():
            tmp_logits = t2n(di_mdl[cell](timg_ii))
        h_ii = tmp_logits.shape[2]
        w_ii = tmp_logits.shape[3]
        # Save for later
        holder_logits[ii,jj,:h_ii,:w_ii] = tmp_logits
        holder_lbls[ii,jj,:h_ii,:w_ii] = cell_ii_bin
    torch.cuda.empty_cache()
    dtime = time() - stime
    nleft, rate = n_sets - (ii+1), (ii+1) / dtime
    seta = (nleft / rate)
    print('ETA: %.1f seconds (%i of %i)' % (seta, ii+1, n_sets))

# Calculate precision/recall for each cell type
holder = []
for tt in list(di_tt_ds):
    for ds in list(di_tt_ds[tt]):
        for jj, cell in enumerate(cells):
            print('~~~ tt=%s, ds=%s, cell=%s ~~~' % (tt, ds, cell))
            idx_tt = df_sets.query('tt==@tt & ds==@ds')['idt'].index.values
            phat_tt = sigmoid(holder_logits[idx_tt, jj, :, :])
            ybin_tt = holder_lbls[idx_tt, jj, :, :]
            tmp_df = global_auprc(ybin_tt, phat_tt, n_points=50)
            tmp_df = tmp_df.assign(cell=cell, tt=tt, ds=ds)
            holder.append(tmp_df)
res_pr = pd.concat(holder).melt(['cell','tt','ds','thresh'],None,'msr')
res_pr['tt'] = pd.Categorical(res_pr['tt'], list(di_tt)).map(di_tt)
res_pr['ds'] = res_pr['ds'].map(di_ds)
res_pr['cell'] = res_pr['cell'].map(di_cell)

# Make a plot
gg_auprc = (pn.ggplot(res_pr, pn.aes(x='thresh',y='value',color='tt',linetype='msr')) + 
    pn.theme_bw() + pn.geom_line() + 
    pn.facet_grid('cell~ds',scales='fixed') + 
    pn.scale_color_discrete(name='Dataset') + 
    pn.scale_linetype_manual(name='Measure',labels=['Precision','Recall'],values=['solid','dashed']) + 
    pn.labs(y='Percent',x='Threshold'))
gg_save('gg_auprc.png', dir_figures, gg_auprc, 11, 5.5)


########################################
## --- (5) PROBABILITY CLUSTERING --- ##

# ONLY VALIDATION SET SHOULD BE USED
# NP.WHERE(IS.NAN(),NAN, ...)

cn_msr = ['r2','rho']
conn_seq = [1, 2]

nquant = 20
idt_train = df_sets.query('tt=="train"')['idt']
idt_val = df_sets.query('tt=="val"')['idt']
idx_train = idt_train.index.values
idx_val = idt_val.index.values
assert len(np.intersect1d(idx_train, idx_val)) == 0, 'Overlap detected!'
# Loop over
holder = []
for jj, cell in enumerate(cells):
    phat_train = sigmoid(holder_logits[idx_train, jj, :, :])
    phat_val = sigmoid(holder_logits[idx_val, jj, :, :])
    ntrain, nrest = len(phat_train), len(phat_val)
    ybin_train = holder_lbls[idx_train, jj, :, :]
    thresh_train = np.quantile(phat_train[ybin_train==1], np.linspace(0, 1, nquant)[1:-1])
    for kk, thresh in enumerate(thresh_train):
        print('~~~~ cell = %s, thresh %i of %i ~~~~' % (cell, kk+1, nquant))
        yhat_train = np.where(phat_train >= thresh, 1, 0)
        yhat_rest = np.where(phat_val >= thresh, 1, 0)
        for conn in conn_seq:
            tmp_train = pd.concat([lbl_freq(yhat_train[i], conn, idt) for i, idt in zip(range(ntrain),idt_train)])
            tmp_rest = pd.concat([lbl_freq(yhat_rest[i], conn, idt) for i, idt in zip(range(nrest),idt_val)])
            # Merge and annotate
            tmp_both = pd.concat([tmp_train, tmp_rest], axis=0).assign(cell=cell, thresh=thresh, conn=conn)
            holder.append(tmp_both)
# Combine all results
res_cell = pd.concat(holder).reset_index(None, drop=True)
# Merge with dataset type
res_cell.rename(columns={'idx':'idt'}, inplace=True)
res_cell = res_cell.merge(df_ds_tt_idt[df_ds_tt_idt['tt'] != 'test'])

# Loop over each combination and the relationship between the n cut-off
q_seq = np.arange(0.05,1,0.05)
holder = []
for jj, cell in enumerate(cells):
    tmp_df1 = res_cell.query('cell==@cell')
    thresh_train = tmp_df1.thresh.unique()
    for kk, thresh in enumerate(thresh_train):
        for conn in conn_seq:
            print('cell=%s, thresh=%.5f, conn=%i' % (cell, thresh, conn))
            tmp_df2 = tmp_df1.query('conn==@conn & thresh==@thresh').reset_index(None, drop=True)
            n_seq = np.sort(tmp_df2.query('tt=="val"')['n'].unique())
            if len(n_seq) > len(q_seq):
                n_seq = np.quantile(n_seq, q_seq)
            for n in n_seq:
                tmp_n = tmp_df2.groupby('idt').apply(lambda x: np.sum(x.n >= n)).reset_index()
                tmp_n = tmp_n.rename(columns={0:'est'}).assign(cell=cell)
                tmp_n = tmp_n.merge(df_cells_long).merge(df_ds_tt_idt)
                tmp_n = tmp_n.groupby(['tt']).apply(lambda x: pd.Series({'rho':rho(x.act, x.est),'r2':r2_score(x.act, x.est)}))
                tmp_n = tmp_n.reset_index().assign(cell=cell,thresh=kk,conn=conn,n=n)
                holder.append(tmp_n)
# Merge and visually inspect trade-off
res_rho = pd.concat(holder).reset_index(None,drop=True)
res_rho['tt'] = res_rho['tt'].map(di_tt)
res_rho['ds'] = res_rho['ds'].map(di_ds)
res_rho['cell'] = res_rho['cell'].map(di_cell)
# Melt on rho vs r2
cn_rho = ['tt','cell','thresh','conn','n']
res_rho = res_rho.melt(cn_rho, cn_msr, 'msr')
# Clip the r-squared
res_rho['value'] = res_rho['value'].clip(-1,1)
res_rho['n2'] = 0
# Convert the min-number into a rank as well for easier plotting
cn_n = ['cell','thresh','msr','conn']
res_rho['gg'] = res_rho[cn_n].astype(str).apply(lambda x: x.str.cat(sep='-'),1)
assert res_rho.groupby('gg').apply(lambda x: x.n.unique().shape[0]).var() == 0, 'Unexpected number of thresholds'
tmp_di_n = res_rho.groupby('gg')['n'].apply(lambda x: x.unique()).to_dict()
tmp_di_n = {k:dict(zip(v,range(len(v)))) for k,v in tmp_di_n.items()}
for k in tmp_di_n:
    tmp_n2 = res_rho[res_rho['gg']==k]['n']
    res_rho.loc[res_rho['gg']==k,'n2'] = tmp_n2.map(tmp_di_n[k])

# Find the "best" point for each cell/dataset
res_rho_star = res_rho.loc[res_rho.groupby(['msr','tt','cell','thresh'])['value'].idxmax()]
res_rho_star.reset_index(None, drop=True, inplace=True)

# Performance as a function of n and thresh --- #
for msr in cn_msr:
    tmp_df = res_rho.query('msr == @msr & tt=="Val"').dropna().reset_index(None, drop=True)
    tmp_df.rename(columns={'thresh':'Thresh_Rank'}, inplace=True)
    tmp_star = res_rho_star.query('msr == @msr & tt=="Val"')
    tmp_star.rename(columns={'thresh':'Thresh_Rank'}, inplace=True)
    tmp_hlines = tmp_star.groupby('cell').value.max().reset_index()
    tmp_fn = 'gg_thresh_n_' + msr + '.png'
    tmp_gg = (pn.ggplot(tmp_df, pn.aes(x='n2',y='value',color='cell',linetype='conn.astype(str)',group='gg')) + 
        pn.theme_bw() + pn.geom_line() + 
        pn.scale_y_continuous(limits=[-1,1]) + 
        pn.facet_wrap('~Thresh_Rank',ncol=6,labeller=pn.labeller(Thresh_Rank=pn.label_both)) + 
        pn.guides(linetype=False) + 
        pn.scale_color_discrete(name='Dataset') + 
        pn.geom_hline(yintercept=0) + 
        pn.geom_hline(pn.aes(yintercept='value',color='cell'),data=tmp_hlines,linetype='--') + 
        pn.ggtitle('Dashed lines show connectivity==2') + 
        pn.geom_point(data=tmp_star,inherit_aes=True,size=2) + 
        pn.theme(subplots_adjust={'wspace': 0.1},legend_position=(0.5,-0.0), legend_direction='horizontal') +
        pn.labs(y='Measure = %s' % msr,x='Minimum cell count rank'))
    gg_save(tmp_fn, dir_figures, tmp_gg, 15, 7.5)


# Plot the overall trade-off
tmp = res_rho_star.query('tt == "Val"').drop(columns=['tt','n'])
tmp['thresh'] = pd.Categorical(tmp['thresh'])
gg_rho_star = (pn.ggplot(tmp,pn.aes(x='thresh',y='n2',color='cell',shape='msr')) + 
    pn.geom_point(position=pn.position_dodge(0.5)) + 
    pn.theme_bw() + pn.scale_color_discrete(name='Cell') + 
    pn.scale_shape_manual(name='Measure',values=['$R$','$ρ$']) + 
    pn.labs(x='Threshold',y='Best rank'))
gg_save('gg_rho_star.png', dir_figures, gg_rho_star, 6, 4)



##############################
## --- (6) FIND OPTIMAL --- ##


di_conn = {'cells':cells, 'thresh':np.zeros(n_cells), 'conn':np.zeros(n_cells), 'n':np.zeros(n_cells)}

holder = []
for jj, cell in enumerate(cells):
    Cell = di_cell[cell]
    tmp_star = res_rho_star.query('cell == @Cell & tt == "Val" & msr=="r2"')
    tmp_star = tmp_star.reset_index(None,drop=True).drop(columns=['cell','tt','msr'])
    tmp_star = tmp_star.query('value == value.max()')
    assert len(tmp_star) == 1
    for cn in ['thresh','conn','n']:
        di_conn[cn][jj] = tmp_star[cn].values[0]

    # Get the scatter plot for the remainder
    sigmoid(holder_logits[:, jj, :, :])

    np.where()

    tmp_star['conn'].values[0]
    int(tmp_star['n'].values[0])
#     phat_val = sigmoid(holder_logits[idx_val, jj, :, :])
#     ybin_val = holder_lbls[idx_val, jj, :, :]
#     thresh_val = np.quantile(phat_val[ybin_val==1], np.linspace(0, 1, nquant)[1:-1])
#     for msr in tmp_star.msr.unique():
#         print('~~~~ cell = %s, msr=%s ~~~~' % (cell, msr))
#         tmp_star_msr = tmp_star.query('msr==@msr').reset_index(None, drop=True)
#         thresh_idx_star = tmp_star_msr.thresh[0]
#         n_star = tmp_star_msr.n[0]
#         conn_star = tmp_star_msr.conn[0]
#         thresh_star = thresh_train[thresh_idx_star]
#         # Apply threshold
#         yhat_train = np.where(phat_train >= thresh_star, 1, 0)
#         yhat_rest = np.where(phat_rest >= thresh_star, 1, 0)
#         # Connect connectivity map
#         tmp_train = pd.concat([lbl_freq(yhat_train[i], conn_star, idt) for i, idt in zip(range(ntrain),idt_train)])
#         tmp_rest = pd.concat([lbl_freq(yhat_rest[i], conn_star, idt) for i, idt in zip(range(nrest),idt_rest)])
#         tmp_both = pd.concat([tmp_train, tmp_rest], axis=0)
#         tmp_both = tmp_both.groupby('idx').apply(lambda x: np.sum(x.n >= n_star))
#         tmp_both = tmp_both.reset_index().rename(columns={'idx':'idt_tissue',0:'est'})
#         tmp_both = tmp_both.assign(cell=cell).merge(df_cells_long).merge(df_ds_tt_idt)
#         tmp_both = tmp_both.assign(thresh=thresh_star, n=n_star, conn=conn_star, msr=msr)
#         holder_n.append(tmp_both)
#         # Use the R-squared model
#         if msr == 'r2':
#             di_conn['thresh'][jj] = thresh_star
#             di_conn['conn'][jj] = conn_star
#             di_conn['n'][jj] = n_star
#             tmp1 = np.stack([phat2lbl(yhat_train[i], thresh_star, n_star, conn_star) for i in range(ntrain)],axis=0)
#             tmp2 = np.stack([phat2lbl(yhat_rest[i], thresh_star, n_star, conn_star) for i in range(nrest)],axis=0)
#             di_conn['post'][:, jj, :, :] = np.vstack([tmp1, tmp2])
#             del tmp1, tmp2
# res_scatter = pd.concat(holder_n).reset_index(None, drop=True)
# res_scatter.tt = pd.Categorical(res_scatter.tt,list(di_tt)).map(di_tt)

# # Save for later
# path_conn = os.path.join(dir_output,'di_conn.pickle')
# hickle.dump(di_conn, path_conn, 'w')

# di_conn = hickle.load(path_conn)
# {k:v.shape for k,v in di_conn.items()}

# https://scikit-image.org/docs/dev/auto_examples/applications/plot_human_mitosis.html#sphx-glr-auto-examples-applications-plot-human-mitosis-py

######################
## --- (7) UNET --- ##

df_unet = np.squeeze(np.apply_over_axes(np.nansum, sigmoid(holder_logits), [2,3]))
df_unet = pd.DataFrame(df_unet / fillfac, columns=cells)
df_unet = pd.concat(objs=[df_sets, df_unet], axis=1)
df_unet = df_unet.melt(['tt','ds','idt'],cells,'cell','est')
df_unet = df_unet.merge(df_cells_long)
df_unet['cell'] = df_unet['cell'].map(di_cell)
df_unet['tt'] = pd.Categorical(df_unet['tt'],list(di_tt)).map(di_tt)

df_unet_rho = df_unet.groupby(['cell','tt']).apply(lambda x: pd.Series({'r2':r2_score(x.act, x.est), 'rho':rho(x.act, x.est)}))
df_unet_rho.reset_index(inplace=True)
df_unet_rho['txt'] = df_unet_rho.apply(lambda x: 'R2=%.1f%%, Rho=%.1f%%' % (x.r2*100,x.rho*100),1)

gg_scatter_unet = (pn.ggplot(df_unet,pn.aes(x='est',y='act',color='tt')) + 
    pn.theme_bw() + pn.geom_point() + 
    pn.scale_color_discrete(name='Dataset') + 
    pn.geom_abline(slope=1,intercept=0,color='black',linetype='--') + 
    pn.facet_wrap('~cell+tt',scales='free',nrow=2) + 
    pn.theme(subplots_adjust={'wspace': 0.20, 'hspace':0.40}) + 
    pn.guides(color=False) + 
    pn.labs(x='Predicted',y='Actual'))
gg_save('gg_scatter_unet.png', dir_figures, gg_scatter_unet, 13, 6)



#########################
## --- (8) FIGURES --- ##

# --- (v) UNet scatter --- #


# # --- (iv) Actual, phat, post-proc --- #
# idt_test = df_ds_tt_idt[df_ds_tt_idt.tt.isin(['test','oos'])].idt_tissue.to_list()
# for kk, idt in enumerate(idt_test):
#     print('image %i of %i (%s)' % (kk+1,len(idt_test),idt))
#     # Find indexes
#     ii = np.where(di_conn['idt_tissue'] == idt)[0][0]
#     ds, tt = df_tt.query('idt_tissue==@idt')[['ds','tt']].values.flat
#     # Load images/labels
#     img_ii = di_data[ds][idt]['img']
#     lbls_ii = np.dstack([di_data[ds][idt]['lbls'][:,:,di_idx[cell]].sum(2) for cell in cells])
#     # Extract phat/yhat
#     logits_ii = di_conn['logits'][ii].transpose(1,2,0)    
#     post_ii = di_conn['post'][ii].transpose(1,2,0)
#     phat_ii = sigmoid(logits_ii)
#     yhat_ii = np.where(phat_ii >= np.expand_dims(di_conn['thresh'],[0,1]),1,0)
#     # Plot it
#     fn_idt = ds + '_' + tt + '_' + idt + '.png'
#     post_plot(img=img_ii, lbls=lbls_ii, phat=phat_ii, yhat=yhat_ii, 
#               fillfac=fillfac, cells=cells, thresh=di_conn['thresh'],
#               fold=dir_inference, fn=fn_idt, title=idt)




# # --- (ii) Visualize the "best" scatter --- #
# gg_scatter_star = (pn.ggplot(res_scatter,pn.aes(x='est',y='act',color='tt')) + 
#     pn.theme_bw() + pn.geom_point() + 
#     pn.scale_color_discrete(name='Dataset') + 
#     pn.geom_abline(slope=1,intercept=0,color='black',linetype='--') + 
#     pn.facet_wrap('~cell+msr+tt',scales='free', nrow=4) + 
#     pn.theme(subplots_adjust={'wspace': 0.20, 'hspace':0.50}) + 
#     pn.guides(color=False) + 
#     pn.labs(x='Predicted',y='Actual'))
# gg_save('gg_scatter_star.png', dir_figures, gg_scatter_star, 13, 12)


# # --- (iii) Point estimate and CI for correlation/R-squared --- #
# n_bs = 250
# alpha = 0.05
# cn_gg = ['cell','tt','msr']
# holder_bs = []
# for i in range(n_bs):
#     for msr in res_scatter.msr.unique():
#         tmp_msr = res_scatter.query('msr == @msr')
#         tmp_df = tmp_msr.groupby(cn_gg).sample(frac=1,replace=True, random_state=i)
#         tmp_df = tmp_df.groupby(cn_gg).apply(lambda x: 
#             pd.Series({'r2':r2_score(x.act, x.est), 'rho':rho(x.act, x.est)}))
#         tmp_df = tmp_df[msr].reset_index().rename(columns={msr:'value'}).assign(bidx=i)
#         holder_bs.append(tmp_df)
# # Merge
# tmp_df = pd.concat(holder_bs).groupby(cn_gg).value.quantile([alpha/2, 1-alpha/2])
# tmp_df = tmp_df.reset_index().pivot_table('value',cn_gg,'level_3')
# tmp_df = tmp_df.reset_index().rename(columns={alpha/2:'lb', 1-alpha/2:'ub'})
# res_cell_perf = res_rho_star.merge(tmp_df)

# # Show the metrics
# posd = pn.position_dodge(0.5)
# gg_perf_star = (pn.ggplot(res_cell_perf,pn.aes(x='msr',y='value',color='cell')) + 
#     pn.theme_bw() + pn.geom_point(position=posd) + 
#     pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
#     pn.scale_color_discrete(name='Cell') + 
#     pn.labs(y='Percent',x='Metric') + 
#     pn.geom_hline(yintercept=0,linetype='--') + 
#     pn.ggtitle('Linerange shows 95% CI') + 
#     pn.theme(subplots_adjust={'wspace': 0.25}) + 
#     pn.facet_wrap('~tt',nrow=2))
# gg_save('gg_perf_star.png', dir_figures, gg_perf_star, 6, 5)

print('~~~ End of 11_explore_test.py ~~~')