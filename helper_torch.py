import numpy as np
import torch
from torch.utils import data
from support_funs_GI import intax3, stopifnot

from scipy.ndimage import rotate


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
            lbls = self.di[id]['lbls'].astype(np.float32)
        else:  # Intergrate out the third access
            lbls = intax3(self.di[id]['lbls']).astype(np.float32)
        imgs = self.di[id]['img'].astype(np.float32) / 255
        if self.transform:
            imgs, lbls = self.transform([imgs, lbls])
        return id, lbls, imgs


class img2tensor(object):
    def __init__(self, device=torch.device('cpu')):
        self.device = device

    def __call__(self, imgs_lbls):
        imgs, lbls = imgs_lbls[0], imgs_lbls[1]
        if not imgs.dtype == np.float32:
            imgs = imgs.astype(np.float32)
        if not lbls.dtype == np.float32:
            lbls = lbls.astype(np.float32)
        # Image in height, width, channels convert to -> c, h, w
        imgs = torch.tensor(imgs.transpose(2, 0, 1)).to(self.device)
        # lbls are in height, width, cells  format -> c, h, w
        lbls = torch.tensor(lbls.transpose(2, 0, 1)).to(self.device)
        return imgs, lbls

class randomRotate(object):
    def __init__(self, tol=1e-4):
        self.tol = tol

    def __call__(self, imgs_lbls):
        imgs, lbls = imgs_lbls[0], imgs_lbls[1]
        k = np.random.randint(4)  # Number of rotations, k==0 is fixed
        angle = k * 90
        # print('Angle: %i' % angle)
        if angle > 0:
            imgs = rotate(imgs, angle, mode='mirror', reshape=False)
            lbls = rotate(lbls, angle, mode='mirror', reshape=False)
            lbls = np.where(lbls <= self.tol, 0, lbls)
        return [imgs, lbls]

class randomFlip(object):
    def __call__(self, imgs_lbls):
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