import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import sys
import config
import os
import torch.nn.functional as F
import odl
from odl.contrib.torch import OperatorModule
import BaseAlg

#####
#####
##### Based on https://arxiv.org/pdf/1704.04058.pdf
#####
#####

M = 5


class update(nn.Module):
    def __init__(self):
        super(update, self).__init__()
        size = config.size
        angles = config.angles


        geometry = odl.tomo.parallel_beam_geometry(config.space, num_angles=angles)
        self.fwd_op = odl.tomo.RayTransform(config.space, geometry, impl='astra_cuda')
        self.grad_op =  odl.Gradient(config.space)
        self.conv = nn.Sequential(
            nn.Conv2d(3+M, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1+M, 3,padding=1),
            nn.BatchNorm2d(1+M),
        )

    def forward(self, f, s, x):
        del_L = OperatorModule(self.fwd_op.adjoint)(OperatorModule(self.fwd_op)(f)-x)
        del_S = OperatorModule(self.grad_op.adjoint)(OperatorModule(self.grad_op)(f))

        output = self.conv(torch.cat([f,s,del_L,del_S],dim=1)*1.0)
        f = f + output[:,M:M+1]
        s = nn.ReLU()(output[:,0:M])
        return (f,s)

class MyNet(nn.Module):
    def __init__(self, args):
        super(MyNet, self).__init__()
        self.update = update()
        self.iterates = args.iterates
        self.hypers={}
    def forward(self, x):
        f = config.fbp_op_mod(x)
        tmp = torch.zeros(f.shape).type_as(f)
        s=tmp.clone()
        for i in range(M-1):
            s=torch.cat((s,tmp),dim=1)
        for i in range(self.iterates):
            (f,s) = self.update(f,s,x)
        return f
    def init_weights(self,m):
        pass
class Algorithm(BaseAlg.baseNet):
    def __init__(self,args,data_loaders,path=config.data_path+'nets/'):
        super(Algorithm, self).__init__(args,path,MyNet(args),data_loaders)
