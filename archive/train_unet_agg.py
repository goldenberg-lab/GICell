import os, pickle
import numpy as np
import pandas as pd
from support_funs import torch2array, sigmoid, ljoin, comp_plt, makeifnot, intax3
from time import time
import torch
from unet_model import UNet
from sklearn import metrics

from helper_torch import CellCounterDataset, img2tensor, randomRotate, randomFlip
from torchvision import transforms
from torch.utils import data

import matplotlib
if not matplotlib.get_backend().lower() == 'agg':
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

######################################
## --- (1) PREP DATA AND MODELS --- ##

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_output, 'figures')
lst_dir = [dir_output, dir_figures]
# assert all([os.path.exists(z) for z in lst_dir])
dir_checkpoint = os.path.join(dir_output, 'checkpoint')
dir_aggregate = os.path.join(dir_checkpoint, 'aggregate')
makeifnot(dir_checkpoint)
makeifnot(dir_aggregate)

# Load data
di_img_point = pickle.load(open(os.path.join(dir_output, 'di_img_point.pickle'), 'rb'))
ids_tissue = list(di_img_point.keys())
print(np.round(di_img_point[ids_tissue[0]]['lbls'].sum() / 9).astype(int))
print(di_img_point[ids_tissue[0]]['pts'].shape[0])

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device('cuda' if use_cuda else 'cpu')

# Get the mean number of cells
pfac = 9  # Number of cells have been multiplied by 9 (basically)
mu_pixels = np.mean([intax3(di_img_point[z]['lbls']).mean() for z in di_img_point])
mu_cells = np.mean([di_img_point[z]['pts'].shape[0] for z in di_img_point])
print('Error: %0.1f' % (mu_pixels*501**2/pfac - mu_cells))
b0 = np.log(mu_pixels / (1 - mu_pixels))

# Load the model
torch.manual_seed(1234)
mdl = UNet(n_channels=3, n_classes=1)
mdl.to(device)
# mdl.load_state_dict(torch.load(os.path.join(dir_ee,'mdl_1000.pt')))
with torch.no_grad():
    mdl.outc.conv.bias.fill_(b0)
# Check CUDA status for model
print('Are network parameters cuda?: %s' %
      all([z.is_cuda for z in mdl.parameters()]))

# Binary loss
criterion = torch.nn.BCEWithLogitsLoss()
# Optimizer
optimizer = torch.optim.Adagrad(params=mdl.parameters(), lr=1e-4)

# tnow = time()
# # Loop through images and make sure they average number of cells equal
# mat = np.zeros([len(ids_tissue), 2])
# for ii, idt in enumerate(ids_tissue):
#     print('ID-tissue %s (%i of %i)' % (idt, ii + 1, len(ids_tissue)))
#     tens = img2tensor(device)([di_img_point[idt]['img'],di_img_point[idt]['lbls']])[0]
#     tens = tens.reshape([1,3,501,501])
#     arr = torch2array(tens)
#     with torch.no_grad():
#         logits = mdl(tens)
#         ncl = logits.cpu().mean().numpy()+0
#         nc = torch.sigmoid(logits).cpu().sum().numpy()+0
#         mat[ii] = [nc, ncl]
# # print(mat.mean(axis=0)); print(mu_pixels*501**2); print(mu_cells); print(b0)
# print('Script took %i seconds' % (time() - tnow))
# torch.cuda.empty_cache()

################################
## --- (2) BEGIN TRAINING --- ##

# Select instances for training/validation
q1 = pd.Series(ids_tissue)
idt_train = q1[~q1.str.contains('MM6IXZVW')].to_list()
idt_val = q1[q1.str.contains('MM6IXZVW')].to_list()
print('%i training samples\n%i validation samples' %
      (len(idt_train), len(idt_val)))

# Create datasetloader class
train_params = {'batch_size': 2,'shuffle': True}
val_params = {'batch_size': len(idt_val),'shuffle': False}
eval_params = {'batch_size': 1,'shuffle': False}

multiclass=False

train_transform = transforms.Compose([randomRotate(tol=1e-4), randomFlip(), img2tensor(device)])
train_data = CellCounterDataset(di=di_img_point, ids=idt_train, transform=train_transform, multiclass=multiclass)
train_gen = data.DataLoader(dataset=train_data,**train_params)
val_transform = transforms.Compose([img2tensor(device)])
val_data = CellCounterDataset(di=di_img_point, ids=idt_val, transform=val_transform, multiclass=multiclass)
val_gen = data.DataLoader(dataset=val_data,**val_params)
eval_data = CellCounterDataset(di=di_img_point, ids=idt_train + idt_val, transform=val_transform, multiclass=multiclass)
eval_gen = data.DataLoader(dataset=eval_data, **eval_params)

tnow = time()
nepochs = 5000
mat_loss = np.zeros([nepochs, 3])
ee, ii = 0, 1
for ee in range(nepochs):
    print('--------- EPOCH %i of %i ----------' % (ee+1, nepochs))
    ii = 0
    np.random.seed(ee)
    torch.manual_seed(ee)
    lst_ce, lst_pred, lst_act = [], [], []
    for ids_batch, lbls_batch, imgs_batch in train_gen:
        ii += 1
        ids_batch = list(ids_batch)
        nbatch = len(ids_batch)
        print('-- batch %i of %i: %s --' % (ii, len(train_gen), ', '.join(ids_batch)))
        # --- Forward pass --- #
        optimizer.zero_grad()
        logits = mdl(imgs_batch)
        assert logits.shape == lbls_batch.shape
        loss = criterion(input=logits,target=lbls_batch)
        # --- Backward pass --- #
        loss.backward()
        # --- Gradient step --- #
        optimizer.step()
        torch.cuda.empty_cache()  # Empty cache
        # --- Performance --- #
        ii_loss = loss.cpu().detach().numpy()+0
        ii_phat = sigmoid(logits.cpu().detach().numpy())
        ii_pred, ii_act = np.zeros(nbatch), np.zeros(nbatch)
        for kk in range(nbatch):
            ii_pred[kk] = ii_phat[kk,0,:,:].sum() / pfac
            ii_act[kk] = di_img_point[ids_batch[kk]]['pts'].shape[0]
        lst_ce.append(ii_loss)
        lst_pred.append(list(ii_pred))
        lst_act.append(list(ii_act))
        print('Cross-entropy loss: %0.4f\nPredicted: %s, actual: %s' %
              (ii_loss, ', '.join(ii_pred.astype(int).astype(str)),
               ', '.join(ii_act.astype(int).astype(str))))
    rho_train = metrics.r2_score(ljoin(lst_act), ljoin(lst_pred))
    ce_train = np.mean(lst_ce)

    # Evaluate model on validation data
    with torch.no_grad():
        for ids_batch, lbls_batch, imgs_batch in val_gen:
            logits = mdl.eval()(imgs_batch)
            ii_phat = sigmoid(logits.cpu().detach().numpy())
            ce_val = criterion(input=logits,target=lbls_batch).cpu().detach().numpy()+0
            nbatch = len(ids_batch)
            ii_pred, ii_act = np.zeros(nbatch), np.zeros(nbatch)
            for kk in range(nbatch):
                ii_pred[kk] = ii_phat[kk, 0, :, :].sum() / pfac
                ii_act[kk] = di_img_point[ids_batch[kk]]['pts'].shape[0]
            print(pd.DataFrame({'id':ids_batch, 'pred':ii_pred.astype(int),'act':ii_act.astype(int)}))
    torch.cuda.empty_cache()  # Empty cache
    print('Cross-entropy - training: %0.4f, validation: %0.4f, cell-count corr: %0.3f' %
          (ce_train, ce_val, rho_train))
    mat_loss[ee] = [ce_train, ce_val, rho_train]
    print('Epoch took %i seconds' % int(time() - tnow))
    tnow = time()
    # Save plots and network every 250 epochs
    if (ee+1) % 250 == 0:
        print('------------ SAVING MODEL AT CHECKPOINT --------------')
        dir_ee = os.path.join(dir_aggregate,'epoch_'+str(ee+1))
        if not os.path.exists(dir_ee):
            os.mkdir(dir_ee)
        with torch.no_grad():
            holder = []
            for ids_batch, lbls_batch, imgs_batch in eval_gen:
                id = ids_batch[0]
                print('Making image for: %s' % id)
                logits = mdl.eval()(imgs_batch).cpu().detach().numpy()
                logits = sigmoid(logits[0,0,:,:])
                img = torch2array(imgs_batch)[:, :, :, 0]
                gaussian = intax3(di_img_point[id]['lbls'])[:,:,0]
                comp_plt(arr=img,pts=logits,gt=gaussian,path=dir_ee,fn=id+'.png',
                         thresh=sigmoid(np.floor(b0)))
                holder.append([id, gaussian.sum() / pfac, logits.sum() / pfac])
        df_ee = pd.DataFrame(holder,columns=['id','act','pred'])
        df_ee['tt'] = np.where(df_ee.id.isin(idt_val), 'Validation', 'Training')
        df_ee.to_csv(os.path.join(dir_ee,'df_'+str(ee+1)+'.csv'),index=False)
        r2_ee = metrics.r2_score(df_ee.act, df_ee.pred)
        # --- make figure --- #
        plt.close()
        g = sns.FacetGrid(df_ee,hue='tt',height=5,aspect=1.2)
        g.map(plt.scatter,'pred','act')
        g.set_xlabels('Predicted')
        g.set_ylabels('Actual')
        g.fig.suptitle('Estimed number of cells at epoch %i\nR-squared: %0.3f' % (ee + 1, r2_ee))
        g.fig.subplots_adjust(top=0.85)
        g.add_legend()
        g._legend.set_title('')
        for ax in g.axes.flatten():
            xmx = int(max(max(ax.get_xlim()),max(ax.get_ylim())))+1
            xmi = int(min(min(ax.get_xlim()),min(ax.get_ylim())))-1
            ax.set_ylim((xmi, xmx))
            ax.set_xlim((xmi, xmx))
            ax.plot([xmi, xmx], [xmi, xmx], '--',c='black')
        g.fig.savefig(os.path.join(dir_ee, 'cell_est.png'))
        # ---- #
        torch.cuda.empty_cache()  # Empty cache
        # Save network
        torch.save(mdl.state_dict(), os.path.join(dir_ee,'mdl_'+str(ee+1)+'.pt'))

# SAVE LOSS AND NETWORK PLEASE!!
df_loss = pd.DataFrame(mat_loss,columns=['train','val','rho'])
df_loss.insert(0,'epoch',np.arange(nepochs)+1)
df_loss = df_loss[df_loss.train != 0].reset_index(None,True)
df_loss.to_csv(os.path.join(dir_output,'mdl_performance.csv'),index=False)
