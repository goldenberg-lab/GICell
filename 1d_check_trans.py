# script to make sure that the rotations/flips are working right as well as the unwind

import os
import imageio
import requests
import torch
import numpy as np
from io import BytesIO
from funs_support import makeifnot, t2n, find_dir_cell
from funs_plotting import plt_single
from funs_torch import img2tensor, randomFlip, randomRotate, all_img_flips

dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
makeifnot(dir_output)
dir_figures = os.path.join(dir_output, 'figures')
makeifnot(dir_figures)
dir_test = os.path.join(dir_figures, 'test')
makeifnot(dir_test)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

###########################
## --- (1) LOAD DATA --- ##

# subtle difference between permute and transpose
# torch.pemute(i, j, k) & np.transpose0 says
#                put i to 0, j to 1, k to 2
xtens = torch.arange(24).reshape([4,3,2])
xarr = np.arange(24).reshape([4,3,2])
assert np.all(t2n(xtens) == xarr)
assert np.all(t2n(xtens.permute([1,2,0])) == xarr.transpose([1,2,0]))

# load sample image
url = 'https://goldenberglab.ca/images/team/erik.jpeg'
res = requests.get(url)
tmp = np.array(imageio.imread(BytesIO(res.content)))
pmax = max(tmp.shape[:2])
img_ed = np.zeros([pmax, pmax, tmp.shape[2]],dtype=int)
img_ed[:tmp.shape[0],:tmp.shape[1]] = tmp
lbl_ed = np.atleast_3d(np.where(img_ed[:,:,2] < 25, 1, 0))
img_lbl_ed = [img_ed, lbl_ed]

# Initialize tensor convert
enc_tens = img2tensor(device,dtype=np.float64)

#####################################
## --- (2) CHECK ROTATION/FLIP --- ##

jj = 0
for k_rotate in range(0,4):
    k_rotate_u = (4 - k_rotate) % 4
    enc_rotate = randomRotate(fix_k=True,k=k_rotate)
    enc_rotate_u = randomRotate(fix_k=True,k=k_rotate_u)
    for k_flip in range(0,3):
        # (i) apply flip/rotate
        k_flip_u = k_flip
        enc_flip = randomFlip(fix_k=True, k=k_flip)
        enc_flip_u = randomFlip(fix_k=True, k=k_flip_u)
        rimg, rlbl = enc_flip(enc_rotate(img_lbl_ed))
        print('rotate=%i, flip=%i, lbl=%i' % (k_rotate, k_flip, rlbl.sum()))
        jj += 1
        # (ii) convert to tensor and back
        tens_rimg, tens_rlbl = enc_tens([rimg, rlbl])
        tens_rimg = t2n(tens_rimg.permute(1,2,0))
        tens_rlbl = t2n(tens_rlbl.permute(1,2,0))

        # (iii) unwind
        rimg_u, rlbl_u = enc_rotate_u(enc_flip_u([tens_rimg, tens_rlbl]))
        assert np.max(np.abs(img_ed - rimg_u)) == 0
        assert np.max(np.abs(lbl_ed - rlbl_u)) == 0
        
        # (iv) plot
        fn = 'rotate_' + str(k_rotate) + '_flip_' + str(k_flip) + '.png'
        plt_single(fn, dir_test, rimg, rlbl)


################################
## --- (3) CHECK FUNCTION --- ##

enc_all = all_img_flips(img_lbl = img_lbl_ed, enc_tens=enc_tens)
enc_all.apply_flips()
enc_all.enc2tensor()
# enc_all.img_tens would need to be put through UNet
lst_rf_img_lbl = enc_tens.tensor2array([enc_all.img_tens, enc_all.lbl_tens])
# Check that reversal works
u_img, u_lbl = enc_all.reverse_flips(lst_rf_img_lbl)
assert np.all(np.expand_dims(img_ed, 3) - u_img == 0)
assert np.all(np.expand_dims(lbl_ed, 3) - u_lbl == 0)
