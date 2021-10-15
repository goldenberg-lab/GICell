# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-np', '--num_param', type=int, default=8,
#                     help='First layer num_params, max=np*(2**4)')
# parser.add_argument('-es', '--epoch_start', type=int, default=0,
#                     help='Whether to preload network at existing epoch')
# parser.add_argument('-ne', '--num_epoch', type=int, default=5000,
#                     help='Maximum number of epochs to run')
# parser.add_argument('-la', '--load_agg', type=bool, default=False,
#                     help='Whether to preload aggregate model network (uses highest epoch)')
# args = parser.parse_args()
# num_param = args.num_param
# epoch_start = args.epoch_start
# num_epoch = args.num_epoch
# load_agg = args.load_agg
# print('Number of params: %i\nEpoch start: %i\nNumber of epochs: %i\nLoad aggregate: %s' %
#       (num_param, epoch_start, num_epoch, load_agg))
# import sys; sys.exit('leaving')

num_param, epoch_start, num_epoch, load_agg = 8, 0, 250, False

import os, pickle
import numpy as np
import pandas as pd
from support_funs_GI import stopifnot, torch2array, sigmoid, comp_plt, sumax3, meanax3, makeifnot, str_subset
from time import time
import torch
from unet_model import UNet
from sklearn import metrics
from scipy.optimize import minimize_scalar

from helper_torch import CellCounterDataset, img2tensor, randomRotate, randomFlip
from torchvision import transforms
from torch.utils import data

import matplotlib
if not matplotlib.get_backend().lower() == 'agg':
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

##############################
## --- (1) PREPARE DATA --- ##

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_output, 'figures')
lst_dir = [dir_output, dir_figures]
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
dir_multiclass = os.path.join(dir_checkpoint, 'multiclass')
dir_aggregate = os.path.join(dir_checkpoint, 'aggregate')
makeifnot(dir_checkpoint)
makeifnot(dir_multiclass)

# Load data
di_img_point = pickle.load(open(os.path.join(dir_output, 'di_img_point.pickle'), 'rb'))
ids_tissue = list(di_img_point.keys())
valid_cells = pd.Series(['eosinophil', 'neutrophil', 'plasma', 'enterocyte', 'other', 'lymphocyte'])
# Sanity check
for idt in ids_tissue:
    tot1 = np.round(di_img_point[idt]['lbls'].sum() / 9).astype(int)
    tot2 = di_img_point[idt]['pts'].shape[0]
    df1 = dict(zip(valid_cells, [np.round(di_img_point[idt]['lbls'][:, :, k].sum() / 9, 0).astype(int) for k
                                 in range(len(valid_cells))]))
    df1 = pd.DataFrame(df1.items(), columns=['cell', 'n'])
    df2 = di_img_point[idt]['pts'].cell.value_counts().reset_index().rename(
        columns={'cell': 'n', 'index': 'cell'})
    checka = tot1 == tot2
    checkb = df2.merge(df1, 'left', 'cell').assign(iseq=lambda x: x.n_x == x.n_y).iseq.all()
    assert checka and checkb), print('ID: %s\nTotal pixels: %i, total points: %i\n%s' % (idt, tot1, tot2, df2.merge(df1, 'left', 'cell')))

# Calculate eosinophilic ratio for each
cn_inflam = ['neutrophil', 'plasma', 'lymphocyte']
cn_eosin = ['eosinophil']
cn_other = ['enterocyte', 'other']
df_eosophils = pd.concat([di_img_point[idt]['pts'].cell.value_counts().reset_index().assign(id=idt) for idt in ids_tissue])
df_eosophils = df_eosophils.pivot('id', 'index', 'cell').fillna(0).reset_index()
df_eosophils.assign(eratio = lambda x: x.eosinophil/(x[cn_inflam].sum()+x.eosinophil))
eratio = df_eosophils.eosinophil / (df_eosophils[cn_inflam].sum(1) + df_eosophils.eosinophil)
eratio = eratio.fillna(0)
df_ratio = pd.DataFrame({'id': df_eosophils.id, 'ratio': eratio}).sort_values('ratio').reset_index(None, True)
agg_cells = ['eosin', 'inflam', 'other']
# Change channels into: eoso's, inflam, other
idx_inflam = np.where(valid_cells.isin(cn_inflam))[0]
idx_eosin = np.where(valid_cells.isin(cn_eosin))[0]
idx_other = np.where(valid_cells.isin(cn_other))[0]
idx_cells = dict(zip(agg_cells, [idx_eosin, idx_inflam, idx_other]))
for idt in ids_tissue:
    tmp = di_img_point[idt]['lbls'].copy()
    tmp2 = np.dstack([tmp[:, :, idx_cells[z]].sum(2) for z in idx_cells])
    di_img_point[idt]['lbls'] = tmp2
    assert di_img_point[idt]['pts'].shape[0] == np.round(di_img_point[idt]['lbls'].sum()/9)
    del tmp, tmp2

###############################
## --- (2) PREPARE MODEL --- ##

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device('cuda' if use_cuda else 'cpu')

# Load the model
pfac = 9  # Number of cells have been multiplied by 9 (basically)
multiclass = True
num_classes = len(agg_cells)
torch.manual_seed(1234)
mdl = UNet(n_channels=3, n_classes=num_classes, bl=num_param)
mdl.to(device)
if epoch_start > 0:
    print('Loading existing model')
    fn_cp = pd.Series(os.listdir())
    fn_cp = fn_cp[fn_cp.str.contains('epoch_'+str(epoch_start))]
    if len(fn_cp) == 1:
        path = os.path.join(dir_multiclass, 'epoch_' + str(epoch_start), 'mdl_'+str(epoch_start)+'.pt')
    mdl.load_state_dict(torch.load(path))
    b0 = mdl.outc.conv.bias.cpu().detach().numpy()
else:
    if load_agg:
        fn_agg = pd.Series(os.listdir(dir_aggregate))
        fn_agg = fn_agg[fn_agg.str.split('_',expand=True).iloc[:,1].astype(int).idxmax()]
        print('Loading aggregate weights: %s' % fn_agg)
        path = os.path.join(dir_aggregate, fn_agg)
        path = os.path.join(path, str_subset(os.listdir(path),'.pt')[0])
        pretrained = torch.load(path)
        pretrained['outc.conv.weight'] = pretrained['outc.conv.weight'].repeat(3, 1, 1, 1)
        pretrained['outc.conv.bias'] = pretrained['outc.conv.bias'].repeat(3)
        mdl.load_state_dict(pretrained, strict=True)
    # Adjust bias
    mat = np.vstack([sumax3(di_img_point[z]['lbls']) for z in di_img_point]).astype(int)
    mu_pixels = mat.mean(0) / (501 ** 2)
    b0 = np.log(mu_pixels / (1 - mu_pixels)).astype(np.float32)
    print('Setting bias to: %s,' % dict(zip(agg_cells, np.round(b0, 2))))
    with torch.no_grad():
        [mdl.outc.conv.bias[k].fill_(b0[k]) for k in range(len(b0))]

def ss2(thresh, logodds, truth, fac):
    return (sigmoid(logodds+thresh).sum()/fac - truth)**2

# # Adjust the intercept where needed
# mat_b0 = np.zeros([len(ids_tissue), num_classes])
# mat_pa = np.zeros([len(ids_tissue), 2*num_classes])
# for ii, idt in enumerate(ids_tissue):
#     print('ID-tissue %s (%i of %i)' % (idt, ii + 1, len(ids_tissue)))
#     tens = img2tensor(device)([di_img_point[idt]['img'], di_img_point[idt]['lbls']])[0]
#     tens = tens.reshape([1]+list(tens.shape))
#     arr = torch2array(tens)
#     gt = (sumax3(di_img_point[idt]['lbls'])/9).astype(int)
#     with torch.no_grad():
#         logits = mdl(tens)
#     logits = logits.cpu().detach().numpy().mean(0).T
#     b0_vec = np.array([minimize_scalar(fun=ss2,bounds=(-5,5),method='bounded',
#            args=(logits[:, :, k],gt[k], pfac)).x for k in range(num_classes)])
#     mat_b0[ii] = b0_vec
#     mat_pa[ii] = np.append(sumax3(sigmoid(b0_vec+logits))/pfac, gt)
# torch.cuda.empty_cache()
# cn_zip = [p+'_'+a for p,a in zip(np.repeat(['pred','act'],num_classes),np.tile(agg_cells,2))]
# df_pa = pd.DataFrame(mat_pa, columns=cn_zip).reset_index().melt('index').assign(variable = lambda x: x.variable.str.split('_'))
# df_pa = pd.concat([df_pa,df_pa.variable.apply(pd.Series).rename(columns={0:'tt',1:'cell'})],1).drop(columns=['variable'])
# df_pa = df_pa.pivot_table('value',['index','cell'],'tt').reset_index().drop(columns=['index'])
# print(np.round(df_pa.groupby('cell').mean().reset_index(),1))
# # Reset the intercept based on the average shift
# b0_adj = mat_b0.mean(0)
# for k, adj in enumerate(b0_adj):
#     b0_k = mdl.outc.conv.bias[k].cpu().detach().numpy()+0
#     with torch.no_grad():
#         mdl.outc.conv.bias[k].fill_(b0_k + adj)
# # update b0
# b0 = mdl.outc.conv.bias.cpu().detach().numpy()

# Check CUDA status for model
print('Are network parameters cuda?: %s' % all([z.is_cuda for z in mdl.parameters()]))

# Optimizer
optimizer = torch.optim.Adagrad(params=mdl.parameters(), lr=5e-4)
# Binary loss
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')  #weight=wmat

##################################
## --- (3) TRAIN UNET MODEL --- ##

# Select validation samples across the quantile spectrum of eosinophil ratio
quantiles = [0.25, 0.50, 0.75]
idt_val = []
for qq in quantiles:
    idx = ((df_ratio.ratio - df_ratio.ratio.quantile(qq)) ** 2).idxmin()
    idt_val.append(df_ratio.loc[idx, 'id'])
    print(df_ratio.iloc[[idx]])

# Select instances for training/validation
q1 = pd.Series(ids_tissue)
idt_train = q1[~q1.isin(idt_val)].to_list()
print('%i training samples\n%i validation samples: %s' %
      (len(idt_train), len(idt_val), ', '.join(idt_val)))

idt_train = ['MM6IXZVW_Cecum']
idt_val = ['MM6IXZVW_Cecum']

# Create datasetloader class
train_params = {'batch_size': 1, 'shuffle': True}
val_params = {'batch_size': len(idt_val), 'shuffle': False}
eval_params = {'batch_size': 1, 'shuffle': False}

val_transform = transforms.Compose([img2tensor(device)])
val_data = CellCounterDataset(di=di_img_point, ids=idt_val, transform=val_transform, multiclass=multiclass)
val_gen = data.DataLoader(dataset=val_data, **val_params)
train_transform = transforms.Compose([randomRotate(tol=1e-4), randomFlip(), img2tensor(device)])
# train_data = CellCounterDataset(di=di_img_point, ids=idt_train, transform=train_transform, multiclass=multiclass)
train_data = CellCounterDataset(di=di_img_point, ids=idt_train, transform=val_transform, multiclass=multiclass)
train_gen = data.DataLoader(dataset=train_data, **train_params)
eval_data = CellCounterDataset(di=di_img_point, ids=idt_train + idt_val, transform=val_transform, multiclass=multiclass)
eval_gen = data.DataLoader(dataset=eval_data, **eval_params)

df_loss = pd.DataFrame(np.zeros([num_epoch-epoch_start, 6]),
                       columns=[p+'_'+a for p,a in zip(np.repeat(['train','val'],num_classes),np.tile(agg_cells,2))])
df_rsq = []
df_comp = []
epoch_iter = np.arange(epoch_start, num_epoch)
tnow = time()
ee, ii = epoch_start, 1  # for debugging
pmod = 50
for ee in epoch_iter:
    np.random.seed(ee)
    torch.manual_seed(ee)
    ii = 0
    # ---- TRAINING SET ----- #
    lst_ce, lst_comp = [], []
    for ids_batch, lbls_batch, imgs_batch in train_gen:
        ii += 1
        ids_batch = list(ids_batch)
        nbatch = len(ids_batch)
        # print('-- batch %i of %i: %s --' % (ii, len(train_gen), ', '.join(ids_batch)))
        # --- Forward pass --- #
        optimizer.zero_grad()
        logits = mdl(imgs_batch)
        assert logits.shape == lbls_batch.shape
        loss = criterion(input=logits, target=lbls_batch)
        # cc = np.random.randint(num_classes)
        # loss = criterion(input=logits[:, cc, :, :], target=lbls_batch[:, cc, :, :])
        # --- Backward pass --- #
        loss.backward()
        # --- Gradient step --- #
        optimizer.step()
        torch.cuda.empty_cache()  # Empty cache
        # --- Performance --- #
        with torch.no_grad():
            ce_train = [criterion(logits[:,k,:,:], lbls_batch[:,k,:,:]) for k in range(logits.shape[1])]
            ce_train = np.array([ce.cpu().detach().numpy()+0 for ce in ce_train])
        ii_phat = sigmoid(logits.cpu().detach().numpy())
        ii_pred = pd.DataFrame({'cell': agg_cells, 'n': sumax3(ii_phat.sum(0).T) / pfac})
        ii_act = pd.DataFrame({'cell': agg_cells,
                                'n': sumax3(lbls_batch.cpu().detach().numpy().sum(0).T) / pfac})
        ii_act = ii_act.merge(ii_pred, 'outer', 'cell', suffixes=('_act', '_pred')).assign(tt='train')
        lst_ce.append(ce_train)
        lst_comp.append(ii_act)
    ii_comp = pd.concat(lst_comp).sort_values('cell').reset_index(None,True)
    ii_mu = ii_comp.groupby('cell').mean().reset_index().melt('cell').assign(epoch = ee+1)
    df_comp.append(ii_mu)
    # rho_train = ii_comp.groupby('cell').apply(lambda x: metrics.r2_score(x.n_act, x.n_pred)).reset_index()
    # rho_train.rename(columns={0: 'rsq'}, inplace=True)
    # df_rsq.append(rho_train.assign(epoch=ee + 1))
    ce_print = ', '.join([c+': {:.4f}'.format(i) for i,c  in zip(ce_train, agg_cells)])
    # rho_print = ', '.join([c + ': {:.4f}'.format(i) for i, c in zip(rho_train.rsq, rho_train.cell)])
    rho_print = 'NaN'
    # ---- VALIDATION SET ----- #
    with torch.no_grad():
        for ids_batch, lbls_batch, imgs_batch in val_gen:
            nbatch = len(ids_batch)
            logits = mdl.eval()(imgs_batch)
            ce_val = [criterion(logits[:, k, :, :], lbls_batch[:, k, :, :]) for k in range(logits.shape[1])]
            ce_val = np.array([ce.cpu().detach().numpy() + 0 for ce in ce_val])
    torch.cuda.empty_cache()  # Empty cache
    ce_print = ', '.join([c + ': {:.4f}'.format(i) for i, c in zip(ce_val, agg_cells)])

    df_loss.iloc[ee] = np.append(ce_train, ce_val)
    if (ee + 1) % pmod == 0:
        print('--------- EPOCH %i of %i ----------' % (ee + 1, num_epoch))
        print(ii_comp)
        print('Training loss: %s\nTraining rho: %s' % (ce_print, rho_print))
        print('Validation loss: %s' % ce_print)
        print('Epoch took %i seconds' % int(time() - tnow))
    tnow = time()
    # Save plots and network every 250 epochs
    if (ee + 1) % pmod == 0:
        print('------------ SAVING MODEL AT CHECKPOINT --------------')
        dir_ee = os.path.join(dir_multiclass, 'epoch_' + str(ee + 1))
        if not os.path.exists(dir_ee):
            os.mkdir(dir_ee)
        # Save network
        torch.save(mdl.state_dict(), os.path.join(dir_ee, 'mdl_' + str(ee + 1) + '.pt'))

        with torch.no_grad():
            holder = []
            for ids_batch, lbls_batch, imgs_batch in eval_gen:
                idt = ids_batch[0]
                print('Making image for: %s' % idt)
                logits = mdl.eval()(imgs_batch).cpu().detach().numpy()
                phat = sigmoid(logits.sum(0).transpose(1, 2, 0))
                img = torch2array(imgs_batch).sum(3)
                gaussian = di_img_point[idt]['lbls'].copy()
                comp_plt(arr=img, pts=phat, gt=gaussian, path=dir_ee, fn=idt + '.png',  #[:,:,[2]]
                         thresh=sigmoid(b0), lbls=agg_cells)  #sigmoid(b0)
                tmp = pd.DataFrame({'id':idt, 'cell':agg_cells,
                                    'act':sumax3(gaussian) / pfac,'pred':sumax3(phat) / pfac})
                holder.append(tmp)
        torch.cuda.empty_cache()  # Empty cache
        df_ee = pd.concat(holder).reset_index(None,True)
        df_ee['tt'] = np.where(df_ee.id.isin(idt_val), 'Validation', 'Training')
        df_ee.to_csv(os.path.join(dir_ee, 'df_' + str(ee + 1) + '.csv'), index=False)
        rho_all = df_ee.groupby('cell').apply(lambda x: metrics.r2_score(x.act, x.pred)).reset_index()
        rho_all.rename(columns={0: 'rsq'}, inplace=True)
        # --- make figure of predicted vs actual --- #
        plt.close()
        g = sns.FacetGrid(df_ee, hue='tt', col='cell',col_wrap=3,
                          sharex=False, sharey=False)  #height=5, aspect=1.2
        g.map(plt.scatter, 'pred', 'act')
        g.set_xlabels('Predicted')
        g.set_ylabels('Actual')
        g.fig.suptitle('Estimed number of cells at epoch %i' % (ee + 1))
        g.fig.subplots_adjust(top=0.85)
        g.add_legend()
        g._legend.set_title('')
        for ax in g.axes.flatten():
            xmx = int(max(max(ax.get_xlim()), max(ax.get_ylim()))) + 1
            xmi = int(min(min(ax.get_xlim()), min(ax.get_ylim()))) - 1
            ax.set_ylim((xmi, xmx))
            ax.set_xlim((xmi, xmx))
            ax.plot([xmi, xmx], [xmi, xmx], '--', c='black')
            cell = ax.get_title().replace('cell = ','')
            rsq = rho_all.loc[rho_all.cell == cell,'rsq'].to_list()[0]
            tt = 'cell: %s , rsq: %0.3f' % (cell, rsq)
            ax.set_title(tt)
        g.fig.savefig(os.path.join(dir_ee, 'cell_est.png'))
        # --- make figure showing the different loss functions --- #
        tmp1 = df_loss.iloc[:ee].assign(epoch=np.arange(0, ee) + 1).melt('epoch',None,'tmp')
        tmp1 = pd.concat([tmp1.drop(columns=['tmp']),tmp1.tmp.str.split('_', expand=True).rename(columns={0:'tt',1:'cell'})],1)
        # tmp1 = pd.concat([tmp1,pd.concat(df_rsq).assign(tt='rsq').rename(columns={'rsq':'value'})],0)
        plt.close()
        g = sns.FacetGrid(tmp1, hue='cell', col='tt', sharex=True, sharey=False)
        g.map(plt.plot,'epoch','value',marker='o')
        g.fig.savefig(os.path.join(dir_ee, 'performance.png'))
        # ---- time series of predicted versus actual --- #
        tmp2 = pd.concat(df_comp)
        plt.close()
        g = sns.FacetGrid(tmp2, hue='variable', row='cell', sharex=True, sharey=False)
        g.map(plt.plot,'epoch','value',marker='o')
        g.add_legend()
        g.fig.savefig(os.path.join(dir_ee, 'pred_act.png'))
        plt.close()


# # SAVE LOSS AND NETWORK PLEASE!!
# df_loss.insert(0, 'epoch', np.arange(num_epoch) + 1)
# df_loss = df_loss[df_loss.train != 0].reset_index(None, True)
# df_loss.to_csv(os.path.join(dir_output, 'mdl_performance.csv'), index=False)
