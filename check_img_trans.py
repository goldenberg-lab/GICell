# script to make sure that the rotations/flips are working right as well as the unwind

import os
from funs_support import makeifnot, t2n, find_dir_cell
from funs_plotting import plt_single
import imageio
import requests
from scipy.ndimage import rotate
from io import BytesIO
import numpy as np
import torch
from funs_torch import img2tensor, randomFlip, randomRotate

dir_base = find_dir_cell()
dir_output = os.path.join(dir_base, 'output')
dir_figures = os.path.join(dir_output, 'figures')
dir_test = os.path.join(dir_figures, 'test')
makeifnot(dir_test)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

################################
## --- (1) CHECK ROTATION --- ##

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
img = np.array(imageio.imread(BytesIO(res.content)))
lbl = np.atleast_3d(np.where(img[:,:,2] < 25, 1, 0))
img_lbl = [img, lbl]

img_all = np.zeros(img.shape + tuple([10]))

# Check rotation
enc_tens = img2tensor(device)
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
        rimg, rlbl = enc_flip(enc_rotate(img_lbl))
        print('rotate=%i, flip=%i, lbl=%i' % (k_rotate, k_flip, rlbl.sum()))
        # !!!!!! re-scale image... !!!!!!!
        img_all[:,:,:,jj] = rimg
        jj += 1
        # (ii) convert to tensor and back
        tens_rimg, tens_rlbl = enc_tens([rimg, rlbl])
        tens_rimg = t2n(tens_rimg.permute(1,2,0))
        tens_rlbl = t2n(tens_rlbl.permute(1,2,0))

        # (iii) unwind
        rimg_u, rlbl_u = enc_rotate_u(enc_flip_u([tens_rimg, tens_rlbl]))
        assert np.max(np.abs(img - rimg_u)) == 0

        # (iv) plot
        fn = 'rotate_' + str(k_rotate) + '_flip_' + str(k_flip) + '.png'
        plt_single(fn, dir_test, rimg, rlbl)

        # fn_u = 'unwind_rotate_' + str(k_rotate) + '_flip_' + str(k_flip) + '.png'
        # plt_single(fn_u, dir_test, rimg_u, rlbl_u)

################################
## --- (2) CHECK FUNCTION --- ##


# img_lbl=img_lbls_ii.copy()
def all_img_flips(img_lbl, device):
    assert len(img_lbl) == 2
    assert img_lbl[0].shape[:2] == img_lbl[1].shape[:2]
    enc_tensor = img2tensor(device)
    enc_tensor(img_lbl)[0].shape
    img_lbl[0]