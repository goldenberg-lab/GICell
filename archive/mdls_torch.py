import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device("cuda" if use_cuda else "cpu")

# https://github.com/NeuroSYS-pl/objects_counting_dmap

channels = [3,6]
size = (3, 3)
stride = 2
N = 1

block = lambda in_channels: nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=channels[1],
                  kernel_size=size,
                  stride=stride,
                  bias=False,
                  padding=(size[0] // 2, size[1] // 2)),
        nn.BatchNorm2d(num_features=channels[1]),
        nn.ReLU()
    )

conv_block(channels, size, stride, N)

class Net(nn.Module):
    def __init__(self,h,w):
        super(Net, self).__init__()
        oc1a, ks1a, st1a = 6, 10, 5
        ks1b, st1b = 5, 5
        h2a = int( (h-ks1a)/st1a +1 )
        w2a = int((w - ks1a) / st1a + 1 )
        h2b = int( (h2a-ks1b)/st1b +1 )
        w2b = int((w2a - ks1b) / st1b + 1)


        self.conv1 = nn.Conv2d(in_channels=3, out_channels=oc1a,
                               kernel_size=ks1a, stride=st1a)
        self.pool1 = nn.MaxPool2d(kernel_size=ks1b,stride=st1b)

        print((h2b, w2b))
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        print(x.shape)
        # return x

torch.manual_seed(1234)
mdl = Net(h=501,w=501)
print(mdl.conv1.weight.data[1,1,0:3,0:3])
tens = torch.tensor(np.random.rand(1,3,501,501).astype(np.float32),
                    device=device)
mdl.eval()(tens)