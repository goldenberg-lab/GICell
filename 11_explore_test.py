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
from funs_stats import global_auroc, global_auprc, rho, lbl_freq, phat2lbl
import plotnine as pn
from mizani.formatters import percent_format
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

cn_msr = ['r2','rho']
di_msr = {'r2':'R-Squared','rho':"Pearson's Rho"}
conn_seq = [1, 2]

nquant = 20
idt_val = df_sets.query('tt=="val"')['idt']
idx_val = idt_val.index.values
n_val = len(idx_val)
# Loop over
holder = []
for jj, cell in enumerate(cells):
    phat_val = sigmoid(holder_logits[idx_val, jj, :, :])
    ybin_val = holder_lbls[idx_val, jj, :, :]
    thresh_val = np.quantile(phat_val[ybin_val==1], np.linspace(0, 1, nquant)[1:-1])
    for kk, thresh in enumerate(thresh_val):
        print('~~~~ cell = %s, thresh %i of %i ~~~~' % (cell, kk+1, nquant))
        # Note that missing values are set to zero which is fine for this case
        yhat_val = np.where(phat_val >= thresh, 1, 0)
        for conn in conn_seq:
            tmp_val = pd.concat([lbl_freq(yhat_val[i], conn, idt) for i, idt in zip(range(n_val),idt_val)]).assign(cell=cell, thresh=thresh, conn=conn)
            holder.append(tmp_val)
# Combine all results
res_cell = pd.concat(holder).reset_index(None, drop=True)
# Merge with dataset type
res_cell.rename(columns={'idx':'idt'}, inplace=True)
# res_cell = res_cell.merge(df_ds_tt_idt[df_ds_tt_idt['tt'] != 'test'])

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
            n_seq = np.sort(tmp_df2['n'].unique())
            if len(n_seq) > len(q_seq):
                n_seq = np.quantile(n_seq, q_seq)
            for n in n_seq:
                tmp_n = tmp_df2.groupby('idt').apply(lambda x: np.sum(x.n >= n)).reset_index()
                tmp_n = tmp_n.rename(columns={0:'est'}).assign(cell=cell)
                tmp_n = tmp_n.merge(df_cells_long)#.merge(df_ds_tt_idt)
                tmp_n = tmp_n.groupby('cell').apply(lambda x: pd.Series({'rho':rho(x.act, x.est),'r2':r2_score(x.act, x.est)}))
                tmp_n = tmp_n.reset_index().assign(thresh2=kk, thresh=thresh,conn=conn,n=n)
                holder.append(tmp_n)
# Merge and visually inspect trade-off
res_rho = pd.concat(holder).reset_index(None,drop=True)
# Melt on rho vs r2
cn_rho = ['cell','thresh','thresh2','conn','n']
res_rho = res_rho.melt(cn_rho, cn_msr, 'msr')
# Clip the r-squared
res_rho['value'] = res_rho['value'].clip(-1)
res_rho['n2'] = 0
# Convert the min-number into a rank as well for easier plotting
cn_n = ['cell','thresh2','msr','conn']
res_rho['gg'] = res_rho[cn_n].astype(str).apply(lambda x: x.str.cat(sep='-'),1)
assert res_rho.groupby('gg').apply(lambda x: x.n.unique().shape[0]).var() == 0, 'Unexpected number of thresholds'
tmp_di_n = res_rho.groupby('gg')['n'].apply(lambda x: x.unique()).to_dict()
tmp_di_n = {k:dict(zip(v,range(len(v)))) for k,v in tmp_di_n.items()}
for k in tmp_di_n:
    tmp_n2 = res_rho[res_rho['gg']==k]['n']
    res_rho.loc[res_rho['gg']==k,'n2'] = tmp_n2.map(tmp_di_n[k])

# Find the "best" point for each cell/dataset
res_rho_star = res_rho.loc[res_rho.groupby(['msr','cell','thresh2'])['value'].idxmax()]
res_rho_star.reset_index(None, drop=True, inplace=True)
res_rho_star.drop(columns='gg',inplace=True)

# Performance as a function of n and thresh --- #
for msr in cn_msr:
    tmp_df = res_rho.query('msr == @msr').dropna().reset_index(None, drop=True)
    tmp_df['cell'] = tmp_df['cell'].map(di_cell)
    tmp_df = tmp_df.rename(columns={'thresh2':'Thresh_Rank'}).drop(columns='msr')
    tmp_star = res_rho_star.query('msr == @msr').rename(columns={'thresh2':'Thresh_Rank'})
    tmp_star['cell'] = tmp_star['cell'].map(di_cell)
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
        pn.geom_point(pn.aes(x='n2',y='value',color='cell'),inherit_aes=False,data=tmp_star,size=2) + 
        pn.theme(subplots_adjust={'wspace': 0.1},legend_position=(0.5,-0.0), legend_direction='horizontal') +
        pn.labs(y='Measure = %s' % msr,x='Minimum cell count rank'))
    gg_save(tmp_fn, dir_figures, tmp_gg, 15, 7.5)


# Plot the overall trade-off
tmp = res_rho_star.drop(columns=['n'])
tmp['thresh2'] = pd.Categorical(tmp['thresh2'])
gg_rho_star = (pn.ggplot(tmp,pn.aes(x='thresh2',y='n2',color='cell',shape='msr')) + 
    pn.geom_point(position=pn.position_dodge(0.5)) + 
    pn.theme_bw() + pn.scale_color_discrete(name='Cell') + 
    pn.scale_shape_manual(name='Measure',values=['$R$','$ρ$']) + 
    pn.labs(x='Threshold',y='Best rank'))
gg_save('gg_rho_star.png', dir_figures, gg_rho_star, 6, 4)


##############################
## --- (6) FIND OPTIMAL --- ##

# ---(i) Create the dictionary to store the "optimal" values for later
di_conn = {'cells':cells, 'thresh':np.zeros(n_cells), 'conn':np.zeros(n_cells), 'n':np.zeros(n_cells)}

for jj, cell in enumerate(cells):
    tmp_star = res_rho_star.query('cell == @cell & msr=="r2"')
    tmp_star = tmp_star.reset_index(None,drop=True).drop(columns=['cell','msr'])
    tmp_star = tmp_star.query('value == value.max()')
    assert len(tmp_star) == 1
    for cn in ['thresh','conn','n']:
        di_conn[cn][jj] = tmp_star[cn].values[0]
# Save
path_conn = os.path.join(dir_output,'di_conn.pickle')
hickle.dump(di_conn, path_conn, 'w')

# --- (ii) Get predicted/actual for all images --- #
holder = []
for ii in range(n_sets):
    if (ii + 1) % 50 == 0:
        print('Iteration %i of %i' % (ii+1, n_sets))
    tmp_ii = df_sets.loc[[ii]].merge(df_cells,'left')
    h_ii = tmp_ii['h'].values[0]
    w_ii = tmp_ii['w'].values[0]
    idt_ii = tmp_ii['idt'].values[0]
    ds_ii = tmp_ii['ds'].values[0]
    tt_ii = tmp_ii['tt'].values[0]
    for jj, cell in enumerate(cells):    
        thresh_jj = di_conn['thresh'][jj]
        conn_jj = di_conn['conn'][jj]
        n_jj = di_conn['n'][jj]
        phat_ii_jj = sigmoid(holder_logits[ii, jj, :, :])
        # Remove missing values
        phat_ii_jj = phat_ii_jj[:h_ii,:w_ii]
        assert np.isnan(phat_ii_jj).sum() == 0, 'Missing values still exist'
        ybin_ii_jj = np.where(phat_ii_jj >= thresh_jj, 1, 0)
        yhat_ii_jj = lbl_freq(ybin_ii_jj, conn_jj, idt_ii)
        yhat_ii_jj = yhat_ii_jj.assign(n=lambda x: np.where(x.n < n_jj, np.nan, x.n))
        yhat_ii_jj = yhat_ii_jj.assign(cell=cell, ds=ds_ii, tt=tt_ii)
        holder.append(yhat_ii_jj)
# Merge and calculate
df_post = pd.concat(holder).rename(columns={'idx':'idt'})
df_post = df_post.groupby(['cell','ds','idt','tt'])['n'].apply(lambda x: x.notnull().sum()).reset_index()
df_post = df_post.rename(columns={'n':'est'}).merge(df_cells_long)
df_post_rho = df_post.groupby(['cell','tt']).apply(lambda x: pd.Series({'r2':r2_score(x.act, x.est), 'rho':rho(x.act, x.est)}))
df_post_rho.reset_index(inplace=True)
df_post_rho['txt'] = df_post_rho.apply(lambda x: 'R2=%.1f%%, Rho=%.1f%%' % (x.r2*100,x.rho*100),1)


######################
## --- (7) UNET --- ##

df_unet = np.squeeze(np.apply_over_axes(np.nansum, sigmoid(holder_logits), [2,3]))
df_unet = pd.DataFrame(df_unet / fillfac, columns=cells)
df_unet = pd.concat(objs=[df_sets, df_unet], axis=1)
df_unet = df_unet.melt(['tt','ds','idt'],cells,'cell','est')
df_unet = df_unet.merge(df_cells_long)

df_unet_rho = df_unet.groupby(['cell','tt']).apply(lambda x: pd.Series({'r2':r2_score(x.act, x.est), 'rho':rho(x.act, x.est)}))
df_unet_rho.reset_index(inplace=True)
df_unet_rho['txt'] = df_unet_rho.apply(lambda x: 'R2=%.1f%%, Rho=%.1f%%' % (x.r2*100,x.rho*100),1)

# Merge with post-hoc and plot
df_both_scatter = pd.concat(objs=[df_post.assign(mdl='Post-Hoc'), df_unet.assign(mdl='U-Net')],axis=0)
df_both_scatter['cell'] = df_both_scatter['cell'].map(di_cell)
df_both_scatter['tt'] = pd.Categorical(df_both_scatter['tt'],list(di_tt)).map(di_tt)
df_both_scatter.reset_index(None, drop=True, inplace=True)

# --- Plot scatter --- #
gg_scatter_unet_post = (pn.ggplot(df_both_scatter,pn.aes(x='np.log(est+1)',y='np.log(act+1)',color='tt')) + 
    pn.theme_bw() + pn.geom_point() + 
    pn.scale_color_discrete(name='Dataset') + 
    pn.geom_abline(slope=1,intercept=0,color='black',linetype='--') + 
    pn.facet_grid('mdl~cell+tt',scales='free') + 
    pn.theme(subplots_adjust={'wspace': 0.1}) + 
    pn.guides(color=False) + 
    pn.labs(x='log(Predicted+1)',y='log(Actual+1)'))
gg_save('gg_scatter_unet_post.png', dir_figures, gg_scatter_unet_post, 12, 5)


###############################
## --- (8) EXTRA FIGURES --- ##

# Repeat for aggregate
df_both_rho = pd.concat(objs=[df_post_rho.assign(mdl='Post-Hoc'), df_unet_rho.assign(mdl='U-Net')],axis=0)
df_both_rho['cell'] = df_both_rho['cell'].map(di_cell)
df_both_rho['tt'] = pd.Categorical(df_both_rho['tt'],list(di_tt)).map(di_tt)
df_both_rho = df_both_rho.melt(['tt','cell','mdl'],cn_msr,'msr')

# --- (i) Uncertainty ranges about R2 + Rho --- #
n_bs = 250
alpha = 0.05
cn_gg = ['cell','tt','mdl']
holder_bs = []
for i in range(n_bs):
    if (i + 1) % 50 == 0:
        print(i+1)
    tmp_bs = df_both_scatter.groupby(cn_gg).sample(frac=1,replace=True,random_state=i)
    tmp_bs = tmp_bs.groupby(cn_gg).apply(lambda x: pd.Series({'r2':r2_score(x.act, x.est), 'rho':rho(x.act, x.est)}))
    holder_bs.append(tmp_bs)
# Merge
tmp_df = pd.concat(holder_bs).reset_index().melt(cn_gg,None,'msr')
tmp_df = tmp_df.groupby(cn_gg+['msr']).value.quantile([alpha/2, 1-alpha/2]).reset_index()
tmp_df = tmp_df.reset_index().pivot_table('value',cn_gg+['msr'],'level_'+str(len(cn_gg)+1))
tmp_df = tmp_df.reset_index().rename(columns={alpha/2:'lb', 1-alpha/2:'ub'})
df_both_rho = df_both_rho.merge(tmp_df)

# (ii) Plot R-squared
posd = pn.position_dodge(0.5)
gg_rho_unet_post = (pn.ggplot(df_both_rho,pn.aes(x='tt',y='value',color='mdl')) + 
    pn.geom_point(position=posd) + pn.theme_bw() + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'), position=posd) + 
    pn.ggtitle('For aggregate predicted vs actual') + 
    pn.facet_grid('msr ~ cell',labeller=pn.labeller(msr=di_msr)) + 
    pn.theme(axis_title_x=pn.element_blank()) + 
    pn.scale_color_discrete(name='Model') + 
    pn.scale_y_continuous(labels=percent_format()) + 
    pn.labs(y='Percent'))
gg_save('gg_rho_unet_post.png', dir_figures, gg_rho_unet_post, 6, 5)


# --- (ii) Actual, phat, post-proc --- #
test_rows = df_sets.query('tt == "test"')
for kk, (ii, rr) in enumerate(test_rows.iterrows()):
    ds, idt, tt = rr['ds'], rr['idt'], rr['tt']
    h_ii, w_ii = df_cells.query('ds==@ds & idt==@idt')[['h','w']].values.flat
    print('image %i of %i (%s)' % (kk+1,len(test_rows),idt))
    
    # (i) Extract data
    phat_ii = sigmoid(holder_logits[ii,:,:h_ii,:w_ii])
    yhat_ii = np.stack([phat2lbl(phat=phat_ii[k], thresh=di_conn['thresh'][k], n=di_conn['n'][k], connectivity=di_conn['conn'][k]) for k in range(len(cells))],axis=0)

    img_ii = di_tt_ds[tt][ds][idt]['img']
    lbl_ii = di_tt_ds[tt][ds][idt]['lbls']
    cell_ii = np.zeros(phat_ii.shape)
    for jj, (k,v) in enumerate(di_idx.items()):
        cell_ii[jj] = lbl_ii[:,:,di_idx[k]].sum(2)

    # Transpose to (h,w,c)
    cell_ii = cell_ii.transpose(1,2,0)
    phat_ii = phat_ii.transpose(1,2,0)
    yhat_ii = yhat_ii.transpose(1,2,0)

    # (ii) Plot
    fn_idt = '%s_%s_%s.png' % (ds, tt, idt)
    post_plot(img=img_ii, lbls=cell_ii, phat=phat_ii, yhat=yhat_ii, fillfac=fillfac, fold=dir_inference, fn=fn_idt, cells=cells, thresh=di_conn['thresh'], title=idt)


print('~~~ End of 11_explore_test.py ~~~')