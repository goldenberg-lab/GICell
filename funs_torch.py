from funs_support import intax3, stopifnot, t2n
import numpy as np
import pandas as pd
import torch
from torch.utils import data


class CellCounterDataset(data.Dataset):
    def __init__(self,di,ids=None,transform=None,multiclass=False):
        self.di = di
        self.multiclass = multiclass
        if ids is None:
            self.ids = list(di.keys())
        else:
            stopifnot(len(np.setdiff1d(ids, list(di.keys()))) == 0)
            self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self,idx):
        id = self.ids[idx]
        if self.multiclass:
            lbls = self.di[id]['lbls']
        else:  # Intergrate out the third access
            lbls = intax3(self.di[id]['lbls'])
        imgs = self.di[id]['img'] / 255
        if self.transform:
            imgs, lbls = self.transform([imgs, lbls])
        return id, lbls, imgs


class img2tensor(object):
    def __init__(self, device=None, dtype=None):
        if device is None:
            device = torch.device('cpu')
        self.device = device
        if dtype is None:
            dtype = np.float32
        self.dtype = dtype

    def __call__(self, imgs_lbls):
        imgs, lbls = imgs_lbls[0].copy(), imgs_lbls[1].copy()
        if not imgs.dtype == self.dtype:
            imgs = imgs.astype(self.dtype)
        if not lbls.dtype == self.dtype:
            lbls = lbls.astype(self.dtype)
        # Image in height, width, channels convert to -> c, h, w
        imgs = torch.tensor(imgs.transpose(2, 0, 1)).to(self.device)
        # lbls are in height, width, cells  format -> c, h, w
        lbls = torch.tensor(lbls.transpose(2, 0, 1)).to(self.device)
        return imgs, lbls

    def tensor2array(self, imgs_lbls):
        imgs, lbls = imgs_lbls[0].clone(), imgs_lbls[1].clone()
        lshp = len(imgs.shape)
        assert isinstance(imgs, torch.Tensor) and isinstance(lbls, torch.Tensor)
        assert lshp == len(lbls.shape) and lshp <= 4 and lshp >= 3
        if lshp == 3:
            imgs = t2n(imgs.permute(1, 2, 0))
            lbls = t2n(lbls.permute(1, 2, 0))
        else:
            imgs = t2n(imgs.permute(2, 3, 1, 0))
            lbls = t2n(lbls.permute(2, 3, 1, 0))
        return [imgs, lbls]


class randomRotate(object):
    def __init__(self, fix_k=False, k=0):
        self.fix_k = fix_k
        self.k = k

    def __call__(self, imgs_lbls):
        imgs, lbls = imgs_lbls[0], imgs_lbls[1]
        if self.fix_k:
            k = self.k
        else:
            k = np.random.randint(4)  # Number of rotations, k==0 is fixed
        if k > 0:
            imgs = np.rot90(m=imgs,axes=(0,1),k=k)
            lbls = np.rot90(m=lbls,axes=(0,1),k=k)
        return [imgs, lbls]

class randomFlip(object):
    def __init__(self, fix_k=False, k=0):
        self.fix_k = fix_k
        self.k = k

    def __call__(self, imgs_lbls):
        if self.fix_k:
            k = self.k
        else:
            k = np.random.randint(3)  # flip: 0 (none), 1 (left-right), 2 (up-down)
        # print('Flip: %i' % k)
        if k == 0:
            return imgs_lbls
        imgs, lbls = imgs_lbls[0], imgs_lbls[1]
        if k == 1:
            imgs, lbls = np.fliplr(imgs).copy(), np.fliplr(lbls).copy()
        if k == 2:
            imgs, lbls = np.flipud(imgs).copy(), np.flipud(lbls).copy()
        return [imgs, lbls]


# --- FUNCTION TO APPLY ALL Flip & Rotate PERMUTATIONS ---- #
# Note, enc2tensor needs to unwound with img2tensor.tensor2array
#       before reverse_flips can be called

class all_img_flips():
    def __init__(self, img_lbl, enc_tens=None, tol=1e-10, is_double=False):
        assert len(img_lbl) == 2 and isinstance(img_lbl, list)
        self.img_lbl = img_lbl.copy()
        self.img = img_lbl[0].copy()
        self.lbl = img_lbl[1].copy()
        self.enc_tens = enc_tens
        assert len(self.img.shape) == 3 and len(self.lbl.shape) == 3
        assert self.img.shape[:2] == self.lbl.shape[:2]
        self.h, self.w, self.c = self.img.shape
        self.p = self.lbl.shape[2]  # Number of labels
        assert self.h == self.w
        self.kseq_rotate = range(4)
        self.kseq_flip = range(3)
        self.ktot = len(self.kseq_rotate) * len(self.kseq_flip)
        self.k_df = pd.DataFrame(np.zeros([self.ktot, 2],dtype=int),columns=['rotate','flip'])
        self.tol = tol
        self.is_double = is_double

    def apply_flips(self):
        self.img_holder = np.zeros([self.h, self.w, self.c, self.ktot],dtype=self.img.dtype)
        self.lbl_holder = np.zeros([self.h, self.w, self.p, self.ktot],dtype=self.lbl.dtype)
        jj = 0
        for k_rotate in self.kseq_rotate:
            enc_rotate = randomRotate(fix_k=True,k=k_rotate)
            for k_flip in self.kseq_flip:
                print('rotate=%i, flip=%i' % (k_rotate, k_flip))
                enc_flip = randomFlip(fix_k=True, k=k_flip)
                rfimg, rflbl = enc_flip(enc_rotate(self.img_lbl))
                assert self.img_lbl[0].sum() - rfimg.sum() == 0
                self.img_holder[:,:,:,jj] = rfimg
                self.lbl_holder[:,:,:,jj] = rflbl
                self.k_df.loc[jj] = [k_rotate, k_flip]
                jj += 1
        # Sanity check (rotations/flips should not change total RGB count)
        v1 = np.apply_over_axes(np.sum, self.img_holder, [0,1,2]).flatten().var()
        v2 = np.apply_over_axes(np.sum, self.lbl_holder, [0,1,2]).flatten().var()
        assert v1 < self.tol and v2 < self.tol

    def enc2tensor(self):
        assert self.enc_tens is not None
        self.img_tens = torch.zeros([self.ktot, self.c, self.h, self.w])
        if self.is_double:
            self.img_tens = self.img_tens.double()
        self.img_tens = self.img_tens.to(self.enc_tens.device)
        self.lbl_tens = torch.zeros([self.ktot, self.p, self.h, self.w])
        if self.is_double:
            self.lbl_tens = self.lbl_tens.double()
        self.lbl_tens = self.lbl_tens.to(self.enc_tens.device)
        for jj in range(self.ktot):
            tmp_lst = [self.img_holder[:,:,:,jj], self.lbl_holder[:,:,:,jj]]
            timg, tlbl = self.enc_tens(tmp_lst)
            self.img_tens[jj,:,:,:] = timg
            self.lbl_tens[jj,:,:,:] = tlbl        
        assert np.all(np.abs(t2n(torch.sum(self.img_tens,[1,2,3])) - self.img.sum()) < self.tol)
        assert np.all(np.abs(t2n(torch.sum(self.lbl_tens,[1,2,3])) - self.lbl.sum()) < self.tol)
        
    def reverse_flips(self, img_lbl):
        assert len(img_lbl) == 2 and isinstance(img_lbl, list)
        arr_img, arr_lbl = img_lbl[0].copy(), img_lbl[1].copy()
        assert isinstance(arr_img, np.ndarray) and isinstance(arr_lbl, np.ndarray)
        rimg_holder = np.zeros([self.h, self.w, self.c, self.ktot])
        rlbl_holder = np.zeros([self.h, self.w, self.p, self.ktot])
        assert rimg_holder.shape == arr_img.shape
        assert rlbl_holder.shape == arr_lbl.shape

        jj = 0
        for k_rotate in self.kseq_rotate:
            # rotate sufficient times to offset
            k_rotate = (4 - k_rotate) % 4 
            enc_rotate = randomRotate(fix_k=True,k=k_rotate)
            for k_flip in self.kseq_flip:
                enc_flip = randomFlip(fix_k=True, k=k_flip)
                tmp_lst = [arr_img[:,:,:,jj], arr_lbl[:,:,:,jj]]
                frimg, frlbl = enc_rotate(enc_flip(tmp_lst))
                rimg_holder[:,:,:,jj] = frimg
                rlbl_holder[:,:,:,jj] = frlbl
                jj += 1
        # Return
        return rimg_holder, rlbl_holder