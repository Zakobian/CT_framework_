import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import sys
import os
import config
import BaseAlg

#####
#####
##### Based on https://www.nature.com/articles/nature25988.pdf
#####
#####


class MyNet(nn.Module):
    def __init__(self, args):
        super(MyNet, self).__init__()
        size = config.angles * config.rec_size
        self.hypers={}
        self.fulnet = nn.Sequential(
            nn.Linear(size, config.size**2),
            nn.Tanh(),
            nn.Linear(config.size**2, config.size**2),
            nn.Tanh(),
        )

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 7, padding=3),
            nn.ReLU(),
        )



    def forward(self, img):
        img = img.view(-1, config.angles * config.rec_size)
        output = self.fulnet(img)
        output = output.view(-1, 1, config.size, config.size)
        output = self.convnet(output)
        return output
        
class Algorithm(BaseAlg.baseNet):
    def __init__(self,args,data_loaders,path=config.data_path+'nets/'):
        super(Algorithm, self).__init__(args,path,data_loaders,MyNet(args))
