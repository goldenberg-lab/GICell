"""
1) ASSOCIATIONS BETWEEN DENSITY AND ORDINAL LABELS
2) COMPARISON OF PERFORMANCE BETWEEN PERIODS
"""

import gc
import os
import pickle
import numpy as np
import pandas as pd
from funs_support import makeifnot, sigmoid, val_plt, t2n, find_dir_cell, plt_single
import torch
from torchvision import transforms
from torch.utils import data
from sklearn.metrics import r2_score
from funs_torch import randomRotate, randomFlip, CellCounterDataset, img2tensor
from mdls.unet import find_bl_UNet
from scipy.stats import kruskal
from scipy import stats

from plotnine import *

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device('cuda' if use_cuda else 'cpu')

dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
dir_snapshot = os.path.join(dir_checkpoint, 'snapshot')
lst_dir = [dir_output, dir_figures, dir_checkpoint, dir_snapshot]
assert all([os.path.exists(path) for path in lst_dir])
dir_inference = os.path.join(dir_figures, 'inference')
makeifnot(dir_inference)

# Get the dates from the snapshot folder
fns_snapshot = pd.Series(os.listdir(dir_snapshot))
fns_snapshot = fns_snapshot[fns_snapshot.str.contains('csv$|pt$')]
dates_snapshot = pd.to_datetime(fns_snapshot.str.split('\\.|\\_', 5, True).iloc[:, 2:5].apply(lambda x: '-'.join(x), 1))
dates2 = pd.Series(dates_snapshot.sort_values(ascending=False).unique())
dnew, dold = dates2[0].strftime('%Y_%m_%d'), dates2[len(dates2) - 1].strftime('%Y_%m_%d')
print('The current date is: %s, the oldest is: %s' % (dnew, dold))
# Make folder in inference with the newest date
dir_save = os.path.join(dir_inference, dnew)
makeifnot(dir_save)
dir_anno = os.path.join(dir_save, 'anno')
makeifnot(dir_anno)

# Load the specific cell order
from cells import valid_cells, inflam_cells


############################################
## --- (1) LOAD DATA THE PRED VS ACT  --- ##

cells = ['Eosinophil', 'Inflammatory']
dates = [dold, dnew]

# Load the data sources
holder = []
for cell in cells:
    for date in dates:
        print('Cell: %s, date: %s' % (cell, date))
        fn = os.path.join(dir_snapshot, 'df_' + cell + '_' + date + '.csv')
        tmp = pd.read_csv(fn).assign(cell=cell, date=date)
        holder.append(tmp)
        del tmp
dat_star = pd.concat(holder).rename(columns={'ids': 'id'}).drop(columns='ce').reset_index(None, True)
del holder
dat_star = dat_star.assign(act=lambda x: x.act.astype(int))
# Check that old Validation is equivalent to new Validation (the reverse need not be true)
check = dat_star.pivot_table('tt', ['id', 'cell'], 'date', lambda x: x)
check = check.rename(columns=dict(zip(dates, ['old', 'new']))).reset_index()
assert all(check.query('old=="Validation"').new == 'Validation')
# Get the ID to labels (with the newest)
di_id = dat_star.query('date==@dnew').groupby(['tt', 'id']).size().reset_index()
assert all(di_id[0] == len(cells))
di_id = dict(zip(di_id.id, di_id.tt))
# Get the training/validation IDs
idt_val = [k for k, q in di_id.items() if q == 'Validation']
idt_train = [k for k, q in di_id.items() if q == 'Training']

# Find the overlap
old_val_idt = check.query('cell=="Eosinophil" & old=="Validation"').id.to_list()

# Calculate the "ratio" as well
cn_gg = ['id', 'tt', 'date']
df_best = dat_star.pivot_table(['act', 'pred'], cn_gg, 'cell').reset_index()
df_best = df_best.melt(cn_gg).rename(columns={None: 'gt'})
df_best = df_best.pivot_table('value', cn_gg + ['gt'], 'cell').reset_index()
df_best = df_best.assign(ratio=lambda x: (x.Eosinophil / x.Inflammatory).fillna(0))
df_best = df_best.pivot_table('ratio', cn_gg, 'gt').reset_index()
df_best = pd.concat([dat_star, df_best.assign(cell='Ratio')]).reset_index(None, True)
# Make a log-scale version for the Cells
df_best_long = df_best.melt(cn_gg+['cell'],None,'y','val')
df_best_long = df_best_long.assign(lval=lambda x: np.where(x.cell=='Ratio',x.val, np.log(x.val)))
df_best_long = df_best_long.melt(cn_gg+['cell','y'],None,'msr')
df_best_w = df_best_long.pivot_table('value',list(df_best_long.columns.drop(['value','y'])),'y').reset_index()
df_best_w = df_best_w.sort_values(['msr','date','cell','tt','id']).query('act>-inf').reset_index(None,True)

# Subset to the log scale msr
# tmp = df_best_w.query('cell=="Eosinophil" & msr=="val"')
tmp = df_best_w
gg_best = (ggplot(tmp, aes(x='pred', y='act', color='date')) + theme_bw() +
           geom_point(alpha=0.5) + geom_abline(slope=1, intercept=0, linetype='--') +
           facet_wrap('~cell+tt+msr', scales='free', labeller=label_both, ncol=4) +
           labs(x='Predicted', y='Actual') +
           theme(legend_position='bottom', legend_box_spacing=0.3,
                 subplots_adjust={'wspace': 0.15, 'hspace': 0.45}) +
           scale_color_discrete(name='Date'))
gg_best.save(os.path.join(dir_save, 'gg_scatter_best.png'), height=8, width=14)

tmp = df_best[df_best.id.isin(old_val_idt)].reset_index(None, True).drop(columns='tt')
tmp['idt'] = tmp.id.map(dict(zip(old_val_idt, range(len(old_val_idt)))))
gg_old_idt = (ggplot(tmp, aes(x='pred', y='act', color='date')) + theme_bw() +
              geom_point(alpha=0.5) + geom_abline(slope=1, intercept=0, linetype='--') +
              facet_wrap('~cell', scales='free', labeller=label_both) +
              labs(x='Predicted', y='Actual') +
              theme(legend_position='bottom', legend_box_spacing=0.3,
                    subplots_adjust={'wspace': 0.15}) +
              scale_color_discrete(name='Date') +
              geom_text(aes(x='pred', y='act', label='idt'), size=10) +
              ggtitle('Pred vs actual (validation old date)\nNumbers are the sample ID'))
gg_old_idt.save(os.path.join(dir_save, 'gg_old_idt.png'), height=4, width=8)

# Get the bootstrap uncertainty around each for the R-squared
nboot = 500
cn_bs = ['msr', 'tt', 'cell', 'date']
dat_r2 = df_best_w.groupby(cn_bs).apply(lambda x: stats.pearsonr(x.act, x.pred)[0]**2).reset_index().rename(columns={0: 'r2'})

holder = []
for ii in range(nboot):
    tmp_df = df_best_w.groupby(cn_bs).sample(frac=1, replace=True, random_state=ii).copy().reset_index(None, True)
    tmp_r2 = tmp_df.groupby(cn_bs).apply(lambda x: stats.pearsonr(x.act, x.pred)[0]**2).reset_index()
    tmp_r2 = tmp_r2.rename(columns={0: 'r2'}).assign(sim=ii)
    holder.append(tmp_r2)
bounds = [0.1, 0.9]
dat_bs = pd.concat(holder).groupby(cn_bs).r2.quantile(bounds).reset_index()
dat_bs = dat_bs.pivot_table('r2', cn_bs, 'level_' + str(len(cn_bs)))
dat_bs = dat_bs.rename(columns={bounds[0]: 'lb', bounds[1]: 'ub'}).reset_index()
dat_bs = dat_bs.merge(dat_r2, 'left', cn_bs)

posd = position_dodge(0.5)
gg_bs_r2 = (ggplot(dat_bs.query('msr=="val"'), aes(x='date', y='r2', color='tt')) +
            theme_bw() + geom_point(position=posd) +
            facet_grid('~cell') +
            geom_linerange(aes(ymin='lb', ymax='ub'), position=posd) +
            ggtitle('Vertical lines shows empirical 80% CI') +
            labs(y='Pearson correlation (squared)', x='Model date') +
            scale_y_continuous(limits=[0,1]))
gg_bs_r2.save(os.path.join(dir_save, 'gg_bs_r2.png'), height=4, width=10)

#######################################################
## --- (2) LOAD THE NANCY + ROBARTS ANNOTATIONS  --- ##

# Valid tissue types
tissues = ['Rectum', 'Ascending', 'Sigmoid', 'Transverse', 'Descending', 'Cecum']

# Make sure we can load the GI ordinal data
dir_GI = os.path.join(dir_base, '..', 'GIOrdinal', 'data')
dir_cleaned = os.path.join(dir_GI, 'cleaned')
assert all([os.path.exists(ff) for ff in [dir_GI, dir_cleaned]])

score_thresh = 1
# Merge with the Nancy + Robarts
cn_keep = ['ID', 'tissue', 'file']
cn_nancy = ['CII', 'AIC']
cn_robarts = ['CII', 'LPN', 'NIE']
dat_nancy = pd.read_csv(os.path.join(dir_GI, 'df_lbls_nancy.csv'), usecols=cn_keep + cn_nancy)
dat_robarts = pd.read_csv(os.path.join(dir_GI, 'df_lbls_robarts.csv'), usecols=cn_keep + cn_robarts)
tmp_nancy = dat_nancy.melt('file', cn_nancy, 'metric').assign(tt='nancy')
tmp_robarts = dat_robarts.melt('file', cn_robarts, 'metric').assign(tt='robarts')
dat_NR = pd.concat([tmp_nancy, tmp_robarts]).reset_index(None, True)
dat_NR = dat_NR[dat_NR.value.notnull()].reset_index(None, True)
dat_NR = dat_NR.assign(value=lambda x: np.where(x.value >= score_thresh, score_thresh, x.value).astype(int))
del tmp_nancy, tmp_robarts, dat_nancy, dat_robarts
tmp = dat_NR.file.str.replace('cleaned_|.png', '').str.split('_', 1, True)
dat_NR = dat_NR.drop(columns='file').assign(idt=tmp[0], tissue=tmp[1])
tmp = dat_NR.tissue.str.split('-', 1, True)
dat_NR = dat_NR.assign(tissue=tmp[0], version=np.where(tmp[1].isnull(), 1, 2))
dat_NR.rename(columns={'value': 'score'}, inplace=True)
print(dat_NR.groupby(['tt', 'metric', 'score']).size())

#########################################
## --- (3) COMPARE TO ENTIRE IMAGE --- ##

di_desc = {'25%': 'lb', '50%': 'med', '75%': 'ub'}
di_cell = {'eosin': 'Eosinophil', 'inflam': 'Inflammatory', 'ratio': 'Ratio'}
di_tt = {'nancy': 'Nancy', 'robarts': 'Robarts'}
cn = ['idt', 'tissue', 'version']
# Load in the full image densities
df_fullimg = pd.read_csv(os.path.join(dir_output, 'df_fullimg.csv'))
df_fullimg = df_fullimg.melt(cn, None, 'cell')
# Merge
df_fullimg = dat_NR.merge(df_fullimg, 'inner', cn)
df_fullimg.cell = df_fullimg.cell.map(di_cell)
df_fullimg.tt = df_fullimg.tt.map(di_tt)
df_fullimg.value = np.log(df_fullimg.value)

# Calculate the median/IQR
nr_desc = df_fullimg.groupby(['metric', 'cell', 'tt', 'score']).value.describe()
nr_desc = nr_desc[di_desc].rename(columns=di_desc).reset_index()

tit = 'Distribution of UNet output to Nancy/Robarts score'
gg_nr = (ggplot(df_fullimg, aes(x='score', y='value')) +
         theme_bw() + ggtitle(tit) +
         geom_jitter(size=0.5, alpha=0.25, random_state=1, width=0.15, height=0, color='blue') +
         labs(x='Score level', y='log(Value)') +
         facet_grid('cell~metric+tt', labeller=label_both, scales='free_y') +
         geom_point(aes(y='med'), data=nr_desc, size=2, color='black') +
         geom_linerange(aes(x='score', ymin='lb', ymax='ub'), color='black', size=1, data=nr_desc, inherit_aes=False) +
         scale_x_continuous(breaks=range(score_thresh + 1), limits=[-0.5, score_thresh + 0.5]))
gg_nr.save(os.path.join(dir_save, 'nancyrobart_ratio.png'), height=8, width=12)

# Calculate the p-value
pval = df_fullimg.groupby(['tt', 'metric', 'cell']).apply(
    lambda x: kruskal(x.value[x.score == 0], x.value[x.score == 1])[1])
pval = pval.reset_index().rename(columns={0: 'pval'})
print(np.round(pval, 3))

#################################################
## --- (4) LOAD THE GROUND TRUTH + MODELS  --- ##

# Load data
di_img_point = pickle.load(open(os.path.join(dir_output, 'di_img_point.pickle'), 'rb'))
ids_tissue = list(di_img_point.keys())
# Image to star eosin and inflam
idx_eosin = np.where(pd.Series(valid_cells) == 'eosinophil')[0]
idx_inflam = np.where(pd.Series(valid_cells).isin(inflam_cells))[0]
for ii, idt in enumerate(ids_tissue):
    tmp = di_img_point[idt]['lbls'].copy()
    tmp_eosin = tmp[:, :, idx_eosin].sum(2)
    tmp_inflam = tmp[:, :, idx_inflam].sum(2)
    tmp2 = np.dstack([tmp_eosin, tmp_inflam])
    tmp3 = di_img_point[idt]['pts'].copy()
    tmp3 = tmp3[tmp3.cell.isin(inflam_cells)]
    di_img_point[idt]['lbls'] = tmp2
    gt, est = tmp3.shape[0], tmp_inflam.sum() / 9
    if gt > 0:
        assert np.abs(gt / est - 1) < 0.02
    del tmp, tmp2, tmp3

# Initialize two models
fn_eosin_new, fn_inflam_new = tuple([os.path.join(dir_snapshot, 'mdl_' + cell + '_' + dnew + '.pt') for cell in cells])
fn_eosin_old, fn_inflam_old = tuple([os.path.join(dir_snapshot, 'mdl_' + cell + '_' + dold + '.pt') for cell in cells])
mdl_eosin_new = find_bl_UNet(path=fn_eosin_new, device=device, batchnorm=True, start=32, stop=64, step=8)
mdl_inflam_new = find_bl_UNet(path=fn_inflam_new, device=device, batchnorm=True, start=32, stop=64, step=8)
mdl_eosin_old = find_bl_UNet(path=fn_eosin_old, device=device, batchnorm=True, start=32, stop=32, step=2)
mdl_inflam_old = find_bl_UNet(path=fn_inflam_old, device=device, batchnorm=True, start=32, stop=32, step=2)

###########################################################
## --- (5) EXAMINE PREDICTED PROBABILITIES ON IMAGE  --- ##

# Loop over validation to make predicted/actual plots
mdl_eosin_new.eval()
mdl_inflam_new.eval()
mdl_eosin_old.eval()
mdl_inflam_old.eval()
torch.cuda.empty_cache()

t_seq = np.round(np.append([0.001],np.arange(0.005,0.101,0.005)),3)

# Use so we know if it's "COMP"
idt_val1 = ['R9I7FYRB_Transverse_17', 'RADS40DE_Rectum_13', '8HDFP8K2_Transverse_5',
           '49TJHRED_Descending_46', 'BLROH2RX_Cecum_72', '8ZYY45X6_Sigmoid_19',
           '6EAWUIY4_Rectum_56', 'BCN3OLB3_Descending_79']

holder = []
holder_pr = []
for idt in idt_val:
    print('id: %s' % idt)
    # --- (i) Load image and calculate density/count --- #
    img, gt = di_img_point[idt]['img'].copy(), di_img_point[idt]['lbls'].copy()
    gt_eosin, gt_inflam = gt[:, :, [0]], gt[:, :, [1]]
    timg = torch.tensor(img.transpose(2, 0, 1).astype(np.float32) / 255).to(device)
    timg = timg.reshape([1] + list(timg.shape))
    with torch.no_grad():
        phat_inflam_new = sigmoid(mdl_inflam_new(timg).cpu().detach().numpy().sum(0).transpose(1, 2, 0))
        phat_eosin_new = sigmoid(mdl_eosin_new(timg).cpu().detach().numpy().sum(0).transpose(1, 2, 0))
        phat_inflam_old = sigmoid(mdl_inflam_old(timg).cpu().detach().numpy().sum(0).transpose(1, 2, 0))
        phat_eosin_old = sigmoid(mdl_eosin_old(timg).cpu().detach().numpy().sum(0).transpose(1, 2, 0))
        torch.cuda.empty_cache()

    pred_inflam_new, pred_eosin_new = phat_inflam_new.sum() / 9, phat_eosin_new.sum() / 9
    pred_inflam_old, pred_eosin_old = phat_inflam_old.sum() / 9, phat_eosin_old.sum() / 9
    act_inflam, act_eosin = int(np.round(gt_inflam.sum() / 9, 0)), int(np.round(gt_eosin.sum() / 9, 0))
    print('ID: %s -- pred new: %i, old: %i (%i), eosin new: %i, old: %i (%i)' %
          (idt, pred_inflam_new, pred_inflam_old, act_inflam, pred_eosin_new, pred_eosin_old, act_eosin))

    # --- (ii) PRECISION/RECALL TRADE-OFF CURVE --- #
    idx_cell_inflam = gt_inflam > 0
    idx_cell_eosin = gt_eosin > 0
    idx_cell_other = idx_cell_inflam & ~idx_cell_eosin
    idx_cell_nothing = gt_inflam == 0
    assert np.sum(idx_cell_inflam) == np.sum(idx_cell_eosin) + np.sum(idx_cell_other)
    assert np.sum(idx_cell_inflam) + np.sum(idx_cell_nothing) == np.prod(img.shape[0:2])

    holder_t = []
    for t in t_seq:
        idx_eosin_t_new = phat_eosin_new > t
        idx_eosin_t_old = phat_eosin_old > t
        n_tp_new = idx_cell_eosin[idx_eosin_t_new].sum()
        n_tp_old = idx_cell_eosin[idx_eosin_t_old].sum()
        n_pred_new = idx_eosin_t_new.sum()
        n_pred_old = idx_eosin_t_old.sum()
        n_gt = idx_cell_eosin.sum()
        tmp = pd.DataFrame({'thresh':t,'tt':[dnew, dold],'n_gt':n_gt,
                  'n_tp':[n_tp_new, n_tp_old], 'n_pred':[n_pred_new, n_pred_old]})
        holder_t.append(tmp)
    holder_pr.append(pd.concat(holder_t).reset_index(None,True).assign(idt=idt))

    # --- (iii) MAKE PLOT FOR 1% THRESHOLD --- #
    # Seperate eosin from inflam
    thresh_eosin, thresh_inflam = 0.01, 0.01
    print('Threshold inflam: %0.5f, eosin: %0.5f' % (thresh_inflam, thresh_eosin))
    # Plot annotations
    fn = 'anno_' + idt + '.png'
    plt_single(arr=img,pts=phat_eosin_new, thresh=thresh_eosin,
               fn=fn,title='Eosinophil annotation',path=dir_anno)

    # Plot comparison between models
    gt = np.dstack([gt_eosin, gt_eosin])
    phat_eosin = np.dstack([phat_eosin_old, phat_eosin_new])
    if idt in idt_val1:
        fn = 'comp_' + idt + '.png'
    else:
        fn = 'eosin_' + idt + '.png'
    val_plt(img, phat_eosin, gt, lbls=dates, path=dir_anno,
            thresh=[thresh_eosin, thresh_eosin], fn=fn)

    # --- (iv) CALCULATE POSITIVE PREDICTIVE VALUES AT 1% --- #
    idx_thresh_eosin = phat_eosin_new > thresh_eosin
    pred_eosin1, pred_eosin2 = phat_eosin_new.sum()/9, phat_eosin_new[idx_thresh_eosin].sum()/9
    num_other, num_eosin, num_null = idx_cell_other[idx_thresh_eosin].sum(), idx_cell_eosin[idx_thresh_eosin].sum(), idx_cell_nothing[idx_thresh_eosin].sum()
    tmp = pd.DataFrame({'idt': idt, 'other': num_other, 'eosin': num_eosin,
                        'null': num_null, 'tot': np.sum(idx_thresh_eosin),
                        'pred1': pred_eosin1, 'pred2':pred_eosin2, 'act': act_eosin}, index=[0])
    holder.append(tmp)

# Calculate precision/recall trade-off
dat_pr = pd.concat(holder_pr).reset_index(None,True)
cn_pr = ['thresh','tt']
dat_pr_agg = dat_pr.drop(columns='idt').groupby(cn_pr).sum().reset_index().assign(Precision=lambda x: x.n_tp/x.n_pred, Recall=lambda x: x.n_tp/x.n_gt)
dat_pr_agg = dat_pr_agg.melt(cn_pr, ['Precision','Recall'],'msr')

gg_pr_agg_comp = (ggplot(dat_pr_agg,aes(x='thresh',y='value',color='tt')) +
             theme_bw() + geom_line() +
             facet_wrap('~msr',scales='free_y') +
             scale_color_discrete(name='Model date') +
                  theme(subplots_adjust={'wspace': 0.15}))
gg_pr_agg_comp.save(os.path.join(dir_figures,'gg_pr_agg_comp.png'),width=8,height=4)

tmp = dat_pr_agg.query('tt == @dnew')
gg_pr_agg = (ggplot(tmp,aes(x='thresh',y='value',color='msr')) +
             theme_bw() + geom_line() +
             scale_color_discrete(name='Model date') +
             scale_y_continuous(limits=[0,1]) +
             scale_x_continuous(breaks=t_seq[np.arange(len(t_seq)) % 2 == 0]) +
             theme(legend_position=(0.75,0.5),axis_text_x=element_text(angle=90)) +
             labs(x='Threshold',y='Percent') +
             ggtitle('Precision/Recall at the pixel level (Eosinophil)'))
gg_pr_agg.save(os.path.join(dir_figures,'gg_pr_curve.png'),width=5,height=4)





# # Find correlation between...
df_inf = pd.concat(holder).melt(['idt', 'pred1', 'pred2', 'act', 'tot'], ['other', 'eosin', 'null'], 'cell', 'n').sort_values(['tot', 'idt']).reset_index(None, True)
df_inf = df_inf.assign(ratio=lambda x: (x.n / x.tot).fillna(0))
tmp = df_inf.assign(cell=lambda x: pd.Categorical(x.cell, ['null', 'other', 'eosin']))
tmp.act = pd.Categorical(tmp.act.astype(str), np.sort(tmp.act.unique()).astype(str))
di_cell = dict(zip(['null', 'other', 'eosin'], ['Empty', 'Other Inflam', 'Eosin']))
gg_inf = (ggplot(tmp, aes(x='act', y='ratio', color='cell')) + theme_bw() +
          geom_jitter(height=0,width=0.1) + facet_wrap('~cell', labeller=labeller(cell=di_cell)) +
          ggtitle('Distribution of points > threshold') +
          labs(y='Percent', x='# of actual eosinophils'))
# # scale_color_discrete(name='Cell type',labels=list(di_cell.values()))
gg_inf.save(os.path.join(dir_save, 'inf_fp_ratio.png'), width=12, height=5)

##################################################
## --- (6) EXAMINE THRESHOLDING TRADE-OFFS  --- ##


# ####################################
# ## --- (6) RANDOM CROPS/FLIPS --- ##
#
# # Get the "actual" cell counts
# df_actcell = dat_star.pivot_table('act', ['id', 'tt'], 'cell').astype(int).reset_index()
#
# # Create datasetloader class
# params = {'batch_size': 1, 'shuffle': False}
#
# k_seq_rotate = [0, 1, 2, 3]
# k_seq_flip = [0, 1, 2]
#
# store = []
# for kr in k_seq_rotate:
#     for kf in k_seq_flip:
#         print('Rotation: %i, Flip: %i' % (kr, kf))
#         transformer = transforms.Compose(
#             [randomRotate(fix_k=True, k=kr), randomFlip(fix_k=True, k=kf), img2tensor(device)])
#         dataset = CellCounterDataset(di=di_img_point, ids=list(di_id), transform=transformer, multiclass=False)
#         generator = data.DataLoader(dataset=dataset, **params)
#         holder = []
#         with torch.no_grad():
#             for ids_batch, lbls_batch, imgs_batch in generator:
#                 # Get predictions
#                 phat_eosin = sigmoid(t2n(mdl_eosin_new(imgs_batch)))
#                 phat_inflam = sigmoid(t2n(mdl_inflam_new(imgs_batch)))
#                 num_eosin, num_inflam = phat_eosin.sum(), phat_inflam.sum()
#                 tmp = pd.DataFrame({'ids': ids_batch, 'eosin': num_eosin, 'inflam': num_inflam}, index=[0])
#                 holder.append(tmp)
#                 # Empty cache
#                 del lbls_batch, imgs_batch
#                 torch.cuda.empty_cache()
#         tmp = pd.concat(holder).assign(rotate=kr, flip=kf)
#         store.append(tmp)
#
# # Compare
# cells = ['eosin', 'inflam', 'ratio']
# dat_flip = pd.concat(store).reset_index(None, True).assign(ratio=lambda x: x.eosin / (x.eosin + x.inflam))
# dat_flip[cells[0:2]] = dat_flip[cells[0:2]] / 9  # Normalize by pfac
# tmp1 = dat_flip.melt(['ids', 'rotate', 'flip'], None, 'cell', 'pred')
# tmp2 = df_actcell.assign(ratio=lambda x: x.eosin / (x.eosin + x.inflam)).rename(columns={'id': 'ids'}).melt(
#     ['ids', 'tt'], None, 'cell', 'act')
# dat_flip = tmp1.merge(tmp2, 'left', ['ids', 'cell']).assign(act=lambda x: x.act.fillna(0))
#
# r2_flip = dat_flip.groupby(['tt', 'rotate', 'flip', 'cell']).apply(
#     lambda x: r2_score(x.act, x.pred)).reset_index().rename(columns={0: 'r2'}).assign(
#     rf=lambda x: 'r=' + x.rotate.astype(str) + ', f=' + x.flip.astype(str))
#
# gg_r2flip = (ggplot(r2_flip, aes(x='rf', y='r2', color='tt', group='tt')) +
#              theme_bw() + geom_point() + geom_line() +
#              labs(y='R-squared') + facet_wrap('~cell') +
#              theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90)) +
#              scale_color_discrete(name='Rotate') +
#              ggtitle('Variation in R2 over flips'))
# gg_r2flip.save(os.path.join(dir_save, 'r2flip.png'), width=12, height=6)
#
# # CV over ID
# cv_flip = dat_flip.groupby(['ids', 'tt', 'cell']).pred.apply(
#     lambda x: pd.Series({'mu': x.mean(), 'se': x.std()})).reset_index()
# cv_flip = cv_flip.pivot_table('pred', ['ids', 'tt', 'cell'], 'level_3').reset_index().assign(cv=lambda x: x.se / x.mu)
#
# gg_cv = (ggplot(cv_flip, aes(x='cv', fill='tt')) + theme_bw() +
#          geom_density(alpha=0.5) + facet_wrap('~cell') +
#          labs(x='CV') + ggtitle('Coefficient of Variation over flip'))
# gg_cv.save(os.path.join(dir_save, 'cv_flip.png'), width=6, height=6)
#
# gg_cv = (ggplot(cv_flip, aes(x='mu', y='cv', color='tt')) + theme_bw() +
#          geom_point() + facet_wrap('~cell', scales='free_x') +
#          theme(subplots_adjust={'wspace': 0.25}) +
#          labs(x='Mean prediction', y='CV') + ggtitle('Coefficient of Variation from Random Flips/Rotations'))
# gg_cv.save(os.path.join(dir_save, 'cv_mu.png'), width=8, height=6)

# #############################################
# ## --- (X) COMPARE TO PREVIOUS MODELS  --- ##
#
# # Save models in a dictionary
# di_mdls = {'eosin': {dates[0]: mdl_eosin_new, dates[1]: mdl_eosin_old},
#            'inflam': {dates[0]: mdl_inflam_new, dates[1]: mdl_inflam_old}}
#
# for ii, idt in enumerate(idt_val):
#     print(ii + 1)
#     img, gt = di_img_point[idt]['img'].copy(), di_img_point[idt]['lbls'].copy()
#     gt_eosin, gt_inflam = gt[:, :, [0]], gt[:, :, [1]]
#     timg = torch.tensor(img.transpose(2, 0, 1).astype(np.float32) / 255).to(device)
#     timg = timg.reshape([1] + list(timg.shape))
#     holder_phat, holder_gt, lbls = [], [], []
#     for cell in di_mdls:
#         for date in di_mdls[cell]:
#             lbls.append(cell + '_' + date)
#             if cell == 'eosin':
#                 holder_gt.append(gt_eosin)
#             else:
#                 holder_gt.append(gt_inflam)
#             print('Cell: %s, date: %s' % (cell, date))
#             with torch.no_grad():
#                 logits = di_mdls[cell][date].eval()(timg).cpu().detach().numpy().sum(0).transpose(1, 2, 0)
#                 phat = sigmoid(logits)
#                 holder_phat.append(phat)
#     phat, gt = np.dstack(holder_phat), np.dstack(holder_gt)
#     assert phat.shape == gt.shape
#     thresholds = list(np.repeat(0.01, len(lbls)))
#     val_plt(img, phat, gt, lbls=lbls, path=dir_save, thresh=thresholds, fn='comp_' + idt + '.png')
