# Script to see how long model takes
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--p',dest='p',type=int,default=64, help='Initial parameter for first channel')
parser.add_argument('--ntrain',dest='ntrain',type=int,default=50)
parser.add_argument('--device',dest='device',type=str,default='cpu', help='Initial parameter for first channel')
args = parser.parse_args()
p, ntrain = args.p, args.ntrain


from funs_support import t2n
import torch
#import numpy as np
from time import time
from mdls.unet import UNet

#import sys; sys.exit('end')

max_channels = p*2**4
print('Baseline: %i, maximum number of channels: %i' % (p, max_channels))
device = torch.device(args.device)
n_channels = 3
pixel_max = 255

mdl = UNet(n_channels=n_channels, n_classes=1, bl=p, batchnorm=True)
mdl.to(device=device)

for i in range(ntrain):
    stime = time()
    img_ii = torch.randn(1, n_channels, pixel_max, pixel_max).to(device=device)
    with torch.no_grad():
        logits_ii = mdl(img_ii)
        logits_ii = logits_ii.cpu().detach().numpy()
        #print(logits_ii.mean())
    dtime = time() - stime
    print('Took %.3f seconds for inference' % dtime)
