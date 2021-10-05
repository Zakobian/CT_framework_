import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import sys
import config
import os
import torch.nn.functional as F
import odl
import BaseAlg
from config import OperatorModule

#####
#####
##### Based on https://arxiv.org/pdf/1707.06474.pdf
#####
#####

n_primal = 5
n_dual = 5


##
## As in the original paper the part of nn represented by the blue block
##
class blue_box(nn.Module):
    def __init__(self):
        super(blue_box, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2+n_primal, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, n_primal, 3,padding=1),
            nn.BatchNorm2d(n_primal),
        )

    def forward(self, h, f, g):
        delf = config.fwd_op_mod(f[:,1:2])
        output = self.conv(torch.cat([h,delf,g],dim=1))
        output = h + output
        return output

##
## As in the original paper the part of nn represented by the red block
##
class red_box(nn.Module):
    def __init__(self):
        super(red_box, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1+n_dual, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, n_dual, 3,padding=1),
            nn.BatchNorm2d(n_dual)
        )

    def forward(self, h, f):
        delh = config.fwd_op_adj_mod(h[:,0:1])
        output = self.conv(torch.cat([f,delh],dim=1))
        output = f + output
        return output


##
## Implementation of the network from the paper
##
class MyNet(nn.Module):
    def __init__(self, args):
        super(MyNet, self).__init__()
        blues = []
        reds = []
        if(args.iterates > 20):
            self.iterates=5
        else:
            self.iterates = args.iterates
        for i in range(self.iterates):
            blues.append(blue_box())
            reds.append(red_box())
        self.blues=torch.nn.ModuleList(blues)
        self.reds=torch.nn.ModuleList(reds)
        self.init_weights(0)

    def forward(self, x):
        h = torch.zeros(x.shape).type_as(x)
        for i in range(n_primal - 1):
            h = torch.cat((h,torch.zeros(x.shape).type_as(x)),dim=1)

        tmp = config.fbp_op_mod(x).clamp(min=0,max=1)
        f = tmp.clone()
        for i in range(n_dual - 1):
            f = torch.cat((f,tmp.clone()),dim=1)

        for i in range(self.iterates):
            h = self.blues[i](h,f,x)
            f = self.reds[i](h,f)
        return f[:,0:1].clamp(min=0,max=1)

        ### 0 bias initialisation
    def init_weights(self,m):
        for layer in range(self.iterates):
            for i in [1,4,7]:
                self.blues[layer].conv[i].bias.data = self.blues[layer].conv[i].bias.data*0.0
                self.reds[layer].conv[i].bias.data = self.reds[layer].conv[i].bias.data*0.0

class Algorithm(BaseAlg.baseNet):
    def __init__(self,args,data_loaders,path=config.data_path+'nets/'):
        super(Algorithm, self).__init__(args,path,MyNet(args),data_loaders)
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=self.args.lr,betas=(0.9,0.99))
