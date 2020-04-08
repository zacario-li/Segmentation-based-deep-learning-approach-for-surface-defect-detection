'''
segmentation-based deep-learning approach for surface-defect detection

author: zacario li
date: 2020-04-08
'''

import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDASDD(nn.Module):
    def __init__(self, numClasses=2):
        super(SDASDD,self).__init__()
        self.numClasses = numClasses
        # segmentation network
        self.sn = SegNetwork(1,1024,1)
        # decision network
        self.dn = DecisionNetwork(self.numClasses)
        
    def forward(self, x):
        out = []
        x = self.sn(x)
        out.append(x[1])
        x = self.dn(x[0], x[1])
        out.append(x)
        return out


class SegNetwork(nn.Module):
    def __init__(self, inChl, outChl1, outChl2):
        super(SegNetwork, self).__init__()
        # 2 x 32
        self.conv1 = nn.Sequential(_Conv2D(inChl, 32, 5),
                                   _Conv2D(32,32,5),
                                   nn.MaxPool2d(2))
        # 3 x 64
        self.conv2 = nn.Sequential(_Conv2D(32,64,5),
                                   _Conv2D(64,64,5),
                                   _Conv2D(64,64,5),
                                   nn.MaxPool2d(2))
        # 4 x 64
        self.conv3 = nn.Sequential(_Conv2D(64,64,5),
                                   _Conv2D(64,64,5),
                                   _Conv2D(64,64,5),
                                   _Conv2D(64,64,5),
                                   nn.MaxPool2d(2))
        # 1 x 1024
        self.conv4 = _Conv2D(64,outChl1,15, padding=7)
        # 1 x 1
        self.conv5 = _Conv2D(1024, outChl2,1, padding=0)
        
    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out.append(x)
        x = self.conv5(x)
        out.append(x)
        return out
        

class DecisionNetwork(nn.Module):
    def __init__(self, numClasses=10):
        super(DecisionNetwork, self).__init__()
        self.numClasses = numClasses
        # convs
        self.convs = nn.Sequential(nn.MaxPool2d(2),
                                   _Conv2D(1025,8,5),
                                   nn.MaxPool2d(2),
                                   _Conv2D(8,16,5),
                                   nn.MaxPool2d(2),
                                   _Conv2D(16,32,5))
        # fc
        self.fc = nn.Linear(66, self.numClasses)
    
    def forward(self, fullmap, segmap):
        # merge two map
        mergeMaps = torch.cat([fullmap, segmap],dim=1)
        # do convs
        out = self.convs(mergeMaps)
        # seg pools
        smpool = F.max_pool2d(segmap,kernel_size=segmap.size()[2:])
        sapool = F.avg_pool2d(segmap,kernel_size=segmap.size()[2:])
        # convs pools
        fmpool = F.max_pool2d(out, kernel_size=out.size()[2:])
        fapool = F.avg_pool2d(out, kernel_size=out.size()[2:])
        # merge pools
        mergeMaps = torch.cat([fmpool, fapool, sapool, smpool],dim=1)
        # fc
        mergeMaps = mergeMaps.view(-1,66)
        out = self.fc(mergeMaps)
        return out
        
class _Conv2D(nn.Module):
    def __init__(self, inChl, outChl, kernel, stride=1, padding=2, **kwargs):
        super(_Conv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChl, outChl, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(outChl),
            nn.ReLU(True)
        )
   
    def forward(self, x):
        x = self.conv(x)
        return x

    
if __name__ == '__main__':
    mdl = SDASDD()
    
    with torch.no_grad():
        x = torch.randn(1,1,512,512)
        out = mdl(x)
        start = time.time()
        for i in range(5):
            print(f'idx : {i}')
            mdl(x)
        print(f'speed is {(time.time() - start)/5} s')
        pass
        