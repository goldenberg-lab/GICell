"""
SCRIPT TO EVALUATE THE MODEL PERFORMANCE ACROSS DIFFERENT EPOCHS
"""

import os, pickle
import numpy as np
import pandas as pd
from funs_support import makeifnot, sigmoid, val_plt, t2n
import torch
from torchvision import transforms
from torch.utils import data
from sklearn.metrics import r2_score
from funs_torch import randomRotate, randomFlip, CellCounterDataset, img2tensor
from funs_unet import find_bl_UNet

# import matplotlib
# if not matplotlib.get_backend().lower() == 'agg':
#     matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# import seaborn as sns

import cv2
from plotnine import *

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device('cuda' if use_cuda else 'cpu')

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_output, 'figures')
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
dir_snapshot = os.path.join(dir_checkpoint, 'snapshot')
lst_dir = [dir_output, dir_figures, dir_checkpoint, dir_snapshot]
assert all([os.path.exists(path) for path in lst_dir])
dir_inference = os.path.join(dir_figures, 'inference')
makeifnot(dir_inference)

# Get the dates from the snapshot folder
fns_snapshot = pd.Series(os.listdir(dir_snapshot))
dates_snapshot = pd.to_datetime(fns_snapshot.str.split('\\.|\\_',5,True).iloc[:,2:5].apply(lambda x: '-'.join(x),1))
dates2 = pd.Series(dates_snapshot.sort_values(ascending=False).unique())
dnew, dold = dates2[0].strftime('%Y_%m_%d'), dates2[len(dates2)-1].strftime('%Y_%m_%d')
print('The current date is: %s, the oldest is: %s' % (dnew, dold))
# Make folder in inference with the newest date
dir_save = os.path.join(dir_inference, dnew)
makeifnot(dir_save)

###########################################
## --- (1) LOAD DATA AND LOAD MODEL  --- ##

# cell order in the lbls matrix
valid_cells = ['eosinophil', 'neutrophil', 'plasma', 'enterocyte', 'other', 'lymphocyte']
inflam_cells = ['eosinophil', 'neutrophil', 'plasma', 'lymphocyte']
# Load data
di_img_point = pickle.load(open(os.path.join(dir_output, 'di_img_point.pickle'), 'rb'))
ids_tissue = list(di_img_point.keys())
# Image to star eosin and inflam
idx_eosin = np.where(pd.Series(valid_cells) == 'eosinophil')[0]
idx_inflam = np.where(pd.Series(valid_cells).isin(inflam_cells))[0]
for idt in ids_tissue:
    tmp = di_img_point[idt]['lbls'].copy()
    tmp_eosin = tmp[:, :, idx_eosin].sum(2)
    tmp_inflam = tmp[:, :, idx_inflam].sum(2)
    tmp2 = np.dstack([tmp_eosin, tmp_inflam])
    tmp3 = di_img_point[idt]['pts'].copy()
    tmp3 = tmp3[tmp3.cell.isin(inflam_cells)]
    di_img_point[idt]['lbls'] = tmp2
    assert np.abs( tmp3.shape[0] - (tmp_inflam.sum() / 9) ) < 1
    del tmp, tmp2, tmp3

cells = ['eosin','inflam']
dates = [dnew, dold]
# Initialize two models
fn_eosin_new, fn_inflam_new = tuple([os.path.join(dir_snapshot, 'mdl_'+cell+'_'+dnew+'.pt') for cell in cells])
fn_eosin_old, fn_inflam_old = tuple([os.path.join(dir_snapshot, 'mdl_'+cell+'_'+dold+'.pt') for cell in cells])
mdl_eosin_new = find_bl_UNet(path=fn_eosin_new, device=device, batchnorm=True)
mdl_inflam_new = find_bl_UNet(path=fn_inflam_new, device=device, batchnorm=True)
mdl_eosin_old = find_bl_UNet(path=fn_eosin_old, device=device, batchnorm=True)
mdl_inflam_old = find_bl_UNet(path=fn_inflam_old, device=device, batchnorm=True)

# Load the data sources
fn_dat_new, fn_dat_old = tuple([os.path.join(dir_snapshot, 'dat_star_'+date+'.csv') for date in dates])
dat_star = pd.read_csv(os.path.join(dir_snapshot, fn_dat_new))
di_id = dat_star.groupby(['tt','id']).size().reset_index()
di_id = dict(zip(di_id.id, di_id.tt))
# Get the training/validation IDs
idt_val = [k for k,q in di_id.items() if q == 'Validation']
idt_train = [k for k,q in di_id.items() if q == 'Training']

# Create Figure with the actual ratio
df_best = dat_star.pivot_table(['act','pred'],['id','tt'],'cell').reset_index()
df_best = df_best.melt(['id','tt']).rename(columns={None:'gt'}).pivot_table('value',['id','tt','gt'],'cell').reset_index()
df_best = df_best.assign(ratio=lambda x: (x.eosin/x.inflam).fillna(0)).pivot_table('ratio',['id','tt'],'gt').reset_index()
df_best = pd.concat([dat_star.drop(columns=['epoch']),df_best.assign(cell='ratio')])

# gg_best = (ggplot(df_best, aes(x='pred',y='act',color='tt')) + theme_bw() +
#            geom_point() + geom_abline(slope=1,intercept=0,linetype='--') +
#            facet_wrap('~cell',scales='free') + labs(x='Predicted',y='Actual') +
#            theme(legend_position='bottom',legend_box_spacing=0.3,
#                  subplots_adjust={'wspace': 0.1}) +
#            scale_color_discrete(name='Set'))
# gg_best.save(os.path.join(dir_save,'gg_scatter_best.png'),height=5,width=12)

###########################################################
## --- (2) EXAMINE PREDICTED PROBABILITIES ON IMAGE  --- ##

# Loop over validation to make predicted/actual plots
mdl_eosin_new.eval()
mdl_inflam_new.eval()
mdl_eosin_old.eval()
mdl_inflam_old.eval()
torch.cuda.empty_cache()

holder = []
for idt in idt_val:
    img, gt = di_img_point[idt]['img'].copy(), di_img_point[idt]['lbls'].copy()
    gt_eosin, gt_inflam = gt[:, :, [0]], gt[:, :, [1]]
    timg = torch.tensor(img.transpose(2, 0, 1).astype(np.float32) / 255).to(device)
    timg = timg.reshape([1] + list(timg.shape))
    with torch.no_grad():
        logits_inflam = mdl_inflam_new(timg).cpu().detach().numpy().sum(0).transpose(1,2,0)
        phat_inflam = sigmoid(logits_inflam)
        logits_eosin = mdl_eosin_new(timg).cpu().detach().numpy().sum(0).transpose(1,2,0)
        phat_eosin = sigmoid(logits_eosin)
    pred_inflam, pred_eosin = phat_inflam.sum()/9, phat_eosin.sum()/9
    act_inflam, act_eosin = int(np.round(gt_inflam.sum()/9,0)), int(np.round(gt_eosin.sum()/9,0))
    print('ID: %s -- pred inflam: %i (%i), eosin: %i (%i)' %
          (idt, pred_inflam, act_inflam, pred_eosin, act_eosin))
    # Seperate eosin from inflam
    thresh_eosin, thresh_inflam = np.quantile(phat_eosin, 0.99), np.quantile(phat_inflam, 0.99)
    print('Threshold inflam: %0.5f, eosin: %0.5f' % (thresh_inflam, thresh_eosin))
    thresh_eosin, thresh_inflam = 0.01, 0.01
    idx_cell_inflam = gt_inflam > 0
    idx_cell_eosin = gt_eosin > 0
    idx_cell_other = idx_cell_inflam & ~idx_cell_eosin
    idx_cell_nothing = gt_inflam == 0
    assert np.sum(idx_cell_inflam) == np.sum(idx_cell_eosin) + np.sum(idx_cell_other)
    assert np.sum(idx_cell_inflam) + np.sum(idx_cell_nothing) == np.prod(img.shape[0:2])
    idx_thresh_eosin = phat_eosin > thresh_eosin
    num_other, num_eosin, num_null = idx_cell_other[idx_thresh_eosin].sum(), idx_cell_eosin[idx_thresh_eosin].sum(), idx_cell_nothing[idx_thresh_eosin].sum()
    tmp = pd.DataFrame({'idt':idt, 'other': num_other, 'eosin': num_eosin,
                  'null': num_null, 'tot': np.sum(idx_thresh_eosin),
                  'pred':pred_eosin, 'act': act_eosin}, index=[0])
    holder.append(tmp)
    gt = np.dstack([gt_eosin, gt_inflam])
    phat = np.dstack([phat_eosin, phat_inflam])
    val_plt(img, phat, gt, lbls=['eosin', 'inflam'], path=dir_save,
             thresh=[thresh_eosin, thresh_inflam], fn=idt+'.png')
# Find correlation between...
df_inf = pd.concat(holder).melt(['idt','pred','act','tot'],['other','eosin','null'],'cell','n').sort_values(['tot','idt']).reset_index(None, True)
df_inf = df_inf.assign(ratio = lambda x: (x.n / x.tot).fillna(0))
tmp = df_inf.assign(cell=lambda x: pd.Categorical(x.cell,['null','other','eosin']))
tmp.act = pd.Categorical(tmp.act.astype(str),np.sort(tmp.act.unique()).astype(str))

di_cell = dict(zip(['null','other','eosin'],['Empty','Other Inflam','Eosin']))
gg_inf = (ggplot(tmp, aes(x='act',y='ratio',color='cell')) + theme_bw() +
          geom_jitter() + facet_wrap('~cell', labeller=labeller(cell=di_cell)) +
          ggtitle('Distribution of points > threshold') +
          labs(y='Percent', x='# of actual eosinophils'))
# scale_color_discrete(name='Cell type',labels=list(di_cell.values()))
gg_inf.save(os.path.join(dir_save,'inf_fp_ratio.png'),width=10,height=5)

#############################################
## --- (3) COMPARE TO PREVIOUS MODELS  --- ##

# Save models in a dictionary
di_mdls = {'eosin':{dates[0]:mdl_eosin_new, dates[1]:mdl_eosin_old},
           'inflam':{dates[0]:mdl_inflam_new, dates[1]:mdl_inflam_old}}

for ii, idt in enumerate(idt_val):
    print(ii+1)
    img, gt = di_img_point[idt]['img'].copy(), di_img_point[idt]['lbls'].copy()
    gt_eosin, gt_inflam = gt[:, :, [0]], gt[:, :, [1]]
    timg = torch.tensor(img.transpose(2, 0, 1).astype(np.float32) / 255).to(device)
    timg = timg.reshape([1] + list(timg.shape))
    holder_phat, holder_gt, lbls = [], [], []
    for cell in di_mdls:
        for date in di_mdls[cell]:
            lbls.append(cell + '_' + date)
            if cell == 'eosin':
                holder_gt.append(gt_eosin)
            else:
                holder_gt.append(gt_inflam)
            print('Cell: %s, date: %s' % (cell, date))
            with torch.no_grad():
                logits = di_mdls[cell][date].eval()(timg).cpu().detach().numpy().sum(0).transpose(1,2,0)
                phat = sigmoid(logits)
                holder_phat.append(phat)
    phat, gt = np.dstack(holder_phat), np.dstack(holder_gt)
    assert phat.shape == gt.shape
    thresholds = list(np.repeat(0.01, len(lbls)))
    val_plt(img, phat, gt, lbls=lbls, path=dir_save, thresh=thresholds, fn='comp_' + idt+'.png')

####################################
## --- (4) RANDOM CROPS/FLIPS --- ##

# Get the "actual" cell counts
df_actcell = dat_star.pivot_table('act',['id','tt'],'cell').astype(int).reset_index()

# Create datasetloader class
params = {'batch_size': 1, 'shuffle': False}

k_seq_rotate = [0, 1, 2, 3]
k_seq_flip = [0, 1, 2]

store = []
for kr in k_seq_rotate:
    for kf in k_seq_flip:
        print('Rotation: %i, Flip: %i' % (kr, kf))
        transformer  = transforms.Compose([randomRotate(fix_k=True, k=kr), randomFlip(fix_k=True,k=kf), img2tensor(device)])
        dataset = CellCounterDataset(di=di_img_point, ids=list(di_id), transform=transformer, multiclass=False)
        generator = data.DataLoader(dataset=dataset, **params)
        holder = []
        with torch.no_grad():
            for ids_batch, lbls_batch, imgs_batch in generator:
                # Get predictions
                phat_eosin = sigmoid(t2n(mdl_eosin_new(imgs_batch)))
                phat_inflam = sigmoid(t2n(mdl_inflam_new(imgs_batch)))
                num_eosin, num_inflam = phat_eosin.sum(), phat_inflam.sum()
                tmp = pd.DataFrame({'ids':ids_batch,'eosin':num_eosin,'inflam':num_inflam},index=[0])
                holder.append(tmp)
                # Empty cache
                del lbls_batch, imgs_batch
                torch.cuda.empty_cache()
        tmp = pd.concat(holder).assign(rotate=kr, flip=kf)
        store.append(tmp)

# Compare
cells = ['eosin','inflam','ratio']
dat_flip = pd.concat(store).reset_index(None,True).assign(ratio=lambda x: x.eosin/(x.eosin+x.inflam))
dat_flip[cells[0:2]] = dat_flip[cells[0:2]] / 9  # Normalize by pfac
tmp1 = dat_flip.melt(['ids','rotate','flip'],None,'cell','pred')
tmp2 = df_actcell.assign(ratio=lambda x: x.eosin/(x.eosin+x.inflam)).rename(columns={'id':'ids'}).melt(['ids','tt'],None,'cell','act')
dat_flip = tmp1.merge(tmp2,'left',['ids','cell']).assign(act=lambda x: x.act.fillna(0))

r2_flip = dat_flip.groupby(['tt','rotate','flip','cell']).apply(lambda x: r2_score(x.act, x.pred)).reset_index().rename(columns={0:'r2'}).assign(rf=lambda x: 'r='+x.rotate.astype(str)+', f='+x.flip.astype(str))

gg_r2flip = (ggplot(r2_flip,aes(x='rf',y='r2',color='tt',group='tt')) +
             theme_bw() + geom_point() + geom_line() +
             labs(y='R-squared') + facet_wrap('~cell') +
             theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90)) +
             scale_color_discrete(name='Rotate') +
             ggtitle('Variation in R2 over flips'))
gg_r2flip.save(os.path.join(dir_save,'r2flip.png'), width=12, height=6)

# CV over ID
cv_flip = dat_flip.groupby(['ids','tt','cell']).pred.apply(lambda x: pd.Series({'mu':x.mean(), 'se':x.std()})).reset_index()
cv_flip = cv_flip.pivot_table('pred',['ids','tt','cell'],'level_3').reset_index().assign(cv=lambda x: x.se/x.mu)

gg_cv = (ggplot(cv_flip, aes(x='cv',fill='tt')) + theme_bw() +
         geom_density(alpha=0.5) + facet_wrap('~cell') +
         labs(x='CV') + ggtitle('Coefficient of Variation over flip'))
gg_cv.save(os.path.join(dir_save,'cv_flip.png'), width=6, height=6)

gg_cv = (ggplot(cv_flip, aes(x='mu',y='cv',color='tt')) + theme_bw() +
         geom_point() + facet_wrap('~cell',scales='free_x') +
         theme(subplots_adjust={'wspace': 0.25}) +
         labs(x='Mean prediction',y='CV') + ggtitle('Coefficient of Variation from Random Flips/Rotations'))
gg_cv.save(os.path.join(dir_save,'cv_mu.png'), width=8, height=6)

#########################################
## --- (5) COMPARE TO ENTIRE IMAGE --- ##

dir_GI = os.path.join(dir_base,'..','..','data')
dir_cleaned = os.path.join(dir_GI, 'cleaned')

k1 = 2000
k2 = int(k1 / 2)
# with torch.no_grad():
#     print(mdl_eosin_new(torch.rand(1,3,kk,kk).to(device)))
#     torch.cuda.empty_cache()

di_desc = {'25%':'lb','50%':'med','75%':'ub'}

cn_keep = ['ID','tissue','file']
cn_nancy = ['CII','AIC']
cn_robarts = ['CII','LPN','NIE']
dat_nancy = pd.read_csv(os.path.join(dir_GI, 'df_lbls_nancy.csv'),usecols=cn_keep+cn_nancy)
dat_robarts = pd.read_csv(os.path.join(dir_GI, 'df_lbls_robarts.csv'),usecols=cn_keep+cn_robarts)
tmp_nancy = dat_nancy.melt('file',cn_nancy,'metric').assign(value=lambda x: np.where(x.value > 3, 2, x.value.fillna(0)).astype(int), tt='nancy')
tmp_robarts = dat_robarts.melt('file',cn_robarts,'metric').assign(value=lambda x: np.where(x.value > 3, 2, x.value.fillna(0)).astype(int), tt='robarts')
dat_NR = pd.concat([tmp_nancy, tmp_robarts]).reset_index(None,True)
dat_IDs = dat_nancy[['ID','tissue','file']]
del dat_nancy, dat_robarts

# Get unique file IDs
mat_num = np.zeros([dat_IDs.shape[0],2])
for ii, rr in dat_IDs.iterrows():
    print('Image %i of %i' % (ii+1, dat_IDs.shape[0]))
    idt, tissue, file = rr['ID'], rr['tissue'], rr['file']
    # Load the image
    path = os.path.join(dir_cleaned,idt, file)
    assert os.path.exists(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    height, width, channels = img.shape
    # Center of image
    i, j = int(np.floor(height / 2)), int(np.floor(width / 2))
    im, ix, jm, jx = 0, k1, 0, k1
    if height > k1:
        im, ix = i - 1000, i + 1000
    if width > k1:
        jm, jx = j - 1000, j + 1000
    print('im: %i, ix: %i, jm: %i, jx: %i' % (im, ix, jm, jx))
    img = img[max(0, i - k2):min(height, i + k2), max(0, j - k2):min(width, j + k2)]
    img = img.reshape(tuple(list(img.shape) + [1])).transpose([3,2,1,0])
    # Convert to GPU tensor
    img = torch.tensor(img / 255, dtype=torch.float32).to(device)
    with torch.no_grad():
        phat_eosin = sigmoid(t2n(mdl_eosin_new(img)))
        torch.cuda.empty_cache()
        phat_inflam = sigmoid(t2n(mdl_inflam_new(img)))
        torch.cuda.empty_cache()
    num_eosin = phat_eosin.sum(0).sum(0).sum(0).sum(0)
    num_inflam = phat_inflam.sum(0).sum(0).sum(0).sum(0)
    mat_num[ii] = [num_eosin, num_inflam]
# Merge and save
dat_num = pd.DataFrame(mat_num, columns=['eosin','inflam'])
dat_num = dat_num.assign(ratio=lambda x: x.eosin / (x.eosin + x.inflam))
dat_num = pd.concat([dat_IDs, dat_num],1)
dat_num.to_csv(os.path.join(dir_output, 'dat_cellcount.csv'),index=False)
# Merge
dat_NR = dat_NR.merge(dat_num[['file','ratio']])
dat_NR.to_csv(os.path.join(dir_output, 'dat_nancyrobarts.csv'),index=False)

# To statistical inference
# nr_long = dat_NR.melt(['ID','tissue','ratio'],['nancy','robarts'],'metric')
nr_desc = dat_NR.groupby(['metric','tt','value']).ratio.describe()[di_desc].rename(columns=di_desc).reset_index()

tit = 'Distribution of eosinophilic ratio to Nancy/Robarts score'
gg_nr = (ggplot(dat_NR, aes(x='value', y='ratio')) +
         theme_bw() + ggtitle(tit) +
         geom_jitter(size=0.5,alpha=0.5,random_state=1,width=0.05,height=0,color='blue') +
         labs(x='Score level', y='Eosinophil ratio') +
         facet_wrap('~tt+metric') +
         geom_point(aes(y='med'),data=nr_desc,size=2,color='black') +
         geom_linerange(aes(x='value',ymin='lb',ymax='ub'),color='black',size=1,data=nr_desc,inherit_aes=False))

gg_nr.save(os.path.join(dir_figures,'nancyrobart_ratio.png'),height=5,width=12)





















