"""
DETERMINE THE SAMPLE SIZE NEEDED FOR A GIVEN CI INTERVAL LENGTH FOR R2
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from plotnine import *
from arch.bootstrap import IIDBootstrap
from time import time
import itertools
import os

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_output, 'figures')

lst_dir = [dir_output, dir_figures]
assert all([os.path.exists(z) for z in lst_dir])

# Generate x, y pairs with specific r2
def dgp_r2(n,r2):
    """
    if y=x+z, x~N(0,1), z~(0,sig2), then rho(y,x)=1/(1+sig2)**0.5, r2(y,x)=1/(1+sig2)
    """
    x = np.random.randn(n)
    sig2 = 1/r2 - 1
    z = np.sqrt(sig2)*np.random.randn(n)
    y = x + z
    return x, y


# Check
x, y = dgp_r2(n=int(1e5), r2=0.33)
print(r2_score(y, x))

#################################
# --- (1) NATURAL VARIATION --- #

nsim = 10000
p_seq = np.arange(0.05,1,0.05)
r2_seq = [0.5] #np.arange(0.3,0.51,0.01)
n_seq = [400] #np.arange(8, 94+1, 1)
df_params = pd.DataFrame(list(itertools.product(r2_seq, n_seq)),columns=['r2','n'])
nparams = df_params.shape[0]

stime = time()
store = []
for ii, rr in df_params.iterrows():
    print('iteration %i of %i' % (ii+1, nparams))
    r2, n = rr['r2'], int(rr['n'])
    holder = np.zeros(nsim)
    np.random.seed(ii)
    for jj in range(nsim):
        if (jj + 1) % 2500 == 0:
            print('Simulation %i of %i' % (jj+1, nsim))
        x, y = dgp_r2(n=n, r2=r2)
        holder[jj] = r2_score(y, x)
        r2_score(y, x)
    tmp = pd.DataFrame({'qq':np.quantile(holder, p_seq),'pp':p_seq, 'n':n, 'r2':r2})
    store.append(tmp)
    rate, nleft = (ii+1)/(time() - stime), nparams-(ii+1)
    print('eta: %i seconds' % (nleft / rate))
# Save
sim_val = pd.concat(store).reset_index(None,True)
sim_val.to_csv(os.path.join(dir_output,'simval.csv'), index=False)
sim_val = pd.read_csv(os.path.join(dir_output,'simval.csv'))
sim_val = sim_val.assign(pp=lambda x: x.pp.round(2).astype(str),
                         r2=lambda x: x.r2.round(2).astype(str))
sim_val = sim_val.pivot_table('qq',['n','r2'],'pp')
sim_val.columns = 'p'+sim_val.columns
sim_val.reset_index(inplace=True)
print(sim_val)

# tmp = sim_val.melt(['n','r2'],['p0.25','p0.5','p0.75'])
# gg_simval = (ggplot(tmp, aes(x='n',y='r2',fill='value')) + theme_bw() +
#              facet_wrap('~pp') + labs(y='R-squared',x='n') +
#              geom_tile(aes(width=1,height=0.01)) +
#              ggtitle('IQR for r2(y,x)') +
#              scale_fill_gradient2(name='R2',low='blue',mid='grey',high='red',midpoint=0.4))
# gg_simval.save(os.path.join(dir_figures,'gg_simvalr2.png'))

### 80% CI AROUND MEDIAN
cn = ['n','r2','p0.1','p0.5','p0.9','spread']
tmp = sim_val[sim_val.r2.isin(['0.3','0.4','0.5'])].assign(spread=lambda x: (x['p0.9']-x['p0.1'])/2)[cn]
gg_simval = (ggplot(tmp, aes(x='n',y='p0.5',color='r2')) + theme_bw() +
             facet_wrap('~r2',labeller=label_both) +
             labs(y='R-squared',x='Sample size (n)') +
             ggtitle('80% CI for r2(y,x)') +
             geom_point() + geom_linerange(aes(ymin='p0.1',ymax='p0.9')) +
             scale_x_continuous(limits=[7,95],breaks=list(np.arange(8,95,8))) +
             guides(color=False))
gg_simval.save(os.path.join(dir_figures,'gg_simvalr2.png'),height=5,width=12)

### AVERAGE INTERVAL LENGTHS
gg_simlenr2 = (ggplot(tmp, aes(x='n',y='spread',color='r2')) + theme_bw() +
             labs(y='Average ± interval lengths',x='Sample size (n)') +
             ggtitle('Spread of 80% CI') + geom_point() +
             scale_y_continuous(limits=[0,0.6],breaks=list(np.arange(0,0.6,0.1))) +
             scale_x_continuous(limits=[7,95],breaks=list(np.arange(8,95,8))))
gg_simlenr2.save(os.path.join(dir_figures,'gg_simlenr2.png'),height=5,width=8)

####################################
# --- (2) CONFIDENCE INTERVALS --- #

if 'powersim.csv' not in os.listdir(dir_output):
    nsim = 1000
    nbs = 249
    r2_seq = np.arange(0.3,0.51,0.01)
    n_seq = np.arange(8, 33, 1)
    df_params = pd.DataFrame(list(itertools.product(r2_seq, n_seq)),columns=['r2','n'])
    nparams = df_params.shape[0]

    stime = time()
    store = []
    for ii, rr in df_params.iterrows():
        print('iteration %i of %i' % (ii+1, nparams))
        r2, n = rr['r2'], int(rr['n'])
        holder = np.zeros([nsim, 3])
        np.random.seed(ii)
        for jj in range(nsim):
            if (jj + 1) % 50 == 0:
                print('Simulation %i of %i' % (jj+1, nsim))
            x, y = dgp_r2(n=n, r2=r2)
            ci = IIDBootstrap(y, x).conf_int(r2_score, reps=nbs, method='bca', size=0.95, tail='two').flatten()
            holder[jj] = np.append(r2_score(y, x), ci)
        tmp = pd.DataFrame(holder, columns=['mu', 'lb', 'ub']).assign(r2=r2, n=n)
        store.append(tmp)
        rate, nleft = (ii+1)/(time() - stime), nparams-(ii+1)
        print('eta: %i seconds' % (nleft / rate))
    # Save for later
    sim_store = pd.concat(store).reset_index(None,True)
    sim_store.to_csv(os.path.join(dir_output,'powersim.csv'), index=False)
else:
    sim_store = pd.read_csv(os.path.join(dir_output,'powersim.csv'))

# Calculate the coverage rates
sim_coverage = sim_store.assign(cover=lambda x: (x.lb < x.r2) & (x.ub > x.r2)).groupby(['r2','n']).cover.mean().reset_index()

gg_coverr2 = (ggplot(sim_coverage,aes(x='n',y='r2',fill='cover')) + theme_bw() +
              geom_tile(aes(width=1,height=0.01)) +
              ggtitle('Coverage from 95% BCa CI for R2') +
              scale_fill_gradient2(name='Coverage',
                                   limits=[0.85,0.951],breaks=list(np.arange(0.85,0.951,0.025)),
                                   low='blue',mid='green',high='yellow',midpoint=0.9))
gg_coverr2.save(os.path.join(dir_figures,'gg_coverr2.png'),height=6,width=8)

# Calculate interval lengths
sim_CIlen = sim_store.assign(spread=lambda x: (x.ub-x.lb)/2).groupby(['r2','n']).spread.mean().reset_index()

gg_spreadr2 = (ggplot(sim_CIlen,aes(x='n',y='r2',fill='spread')) + theme_bw() +
              geom_tile(aes(width=1,height=0.01)) +
              ggtitle('Coverage lengths from 95% BCa CI for R2') +
              scale_fill_gradient2(name='Spread', low='blue', mid='green', high='yellow', midpoint=0.75,
                                   limits=[0.25,1.25],breaks=list(np.arange(0.25,1.25,0.25))))
gg_spreadr2.save(os.path.join(dir_figures,'gg_spreadr2.png'),height=6,width=8)

# scale_fill_gradient2(name='Spread', low='blue', mid='green', high='yellow', midpoint=0.75)
# limits = [0.85, 0.951], breaks = list(np.arange(0.85, 0.951, 0.025))

################################
# --- (3) R2 over training --- #

burnin = 0
epoch_eosin = pd.read_csv(os.path.join(dir_output,'mdl_performance_eosin.csv'))
epoch_inflam = pd.read_csv(os.path.join(dir_output,'mdl_performance_inflam.csv'))
df_epoch = pd.concat([epoch_eosin.assign(cell='eosin'), epoch_inflam.assign(cell='inflam')])
del epoch_eosin, epoch_inflam
df_epoch = df_epoch[(df_epoch.epoch > burnin) & (df_epoch.metric=='r2')].reset_index(None,True)
df_epoch = df_epoch.assign(val2=lambda x: x.val.clip(0))
cn = ['tt','batch','cell']
df_epoch = df_epoch.sort_values(cn+['epoch']).reset_index(None,True)
tmp = df_epoch.groupby(cn).val2.rolling(window=10).mean().reset_index().drop(columns='level_3')
df_epoch = df_epoch.assign(trend=tmp.val2)

gg_epoch = (ggplot(df_epoch, aes(x='epoch',y='trend')) +
            geom_line(aes(linetype='batch',color='tt')) +
            ggtitle('Training/Val performance') +
            facet_grid('cell~') +
            theme_bw())
gg_epoch.save(os.path.join(dir_figures,'gg_epoch.png'),height=9,width=12)

print(df_epoch[df_epoch.epoch==27].iloc[:,0:6])
print(df_epoch[df_epoch.epoch==29].iloc[:,0:6])
df_epoch.groupby('epoch').val.min().reset_index().sort_values('val',ascending=False)


