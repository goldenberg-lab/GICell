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

# Bootstrap interval construction for r2
nsim = 1000
nbs = 249
n = 150
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
    tmp = pd.DataFrame(holder, columns=['r2', 'lb', 'ub']).assign(r2=r2, n=n)
    store.append(tmp)
    rate, nleft = (ii+1)/(time() - stime), nparams-(ii+1)
    print('eta: %i seconds' % (nleft / rate))
# Save for later
sim_store = pd.concat(store).reset_index(None,True)
sim_store.to_csv(os.path.join(dir_output,'powersim.csv'), index=False)
