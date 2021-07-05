import numpy as np
import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA IS AVAILABLE, SETTING DEVICE')
    print('# of CUDA devices: %i' % torch.cuda.device_count())
else:
    print('CUDA IS NOT AVAILABLE, USING CPU')
device = torch.device("cuda" if use_cuda else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=10, stride=1)
    def forward(self, x):
        x = self.conv1(x)
        return x

model = Net()
model.to(device)
print('Are network parameters cuda?: %s' %
      all([z.is_cuda for z in model.parameters()]))

tens1 = torch.tensor(np.random.rand(1,3,50,50).astype(np.float32),
                     device=device)
tens2 = torch.tensor(np.random.rand(1,3,50,50).astype(np.float32))
print('tens1 cuda: %s, output1 cuda: %s\n'
      'tens2 cuda: %s' %  #, output2 cuda: %s
      (tens1.is_cuda, model(tens1).is_cuda, tens2.is_cuda))
if not tens1.is_cuda:
    print('conv is cuda: %s' % model(tens2).is_cuda)
