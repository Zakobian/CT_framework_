from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import traceback
from IPython.display import display, clear_output
import config
from torch.autograd import Variable


##
## Custom class for loading data
##
class MyCustomDataset(Dataset):
    def __init__(self, percent, direc, transform,args):
        self.data_root = direc
        self.args=args
        self.transform = transform
        if (os.path.exists((os.path.join(self.data_root, '/labels')))):
            self.names = np.array([name for name in os.listdir((os.path.join(self.data_root, '/labels'+name)))])
        else: self.names = np.array([name for name in os.listdir(self.data_root)])
        if percent<0:self.names = self.names[0:-1*percent]
        else: self.names = self.names[0:int(percent*len(self.names)//100)]
        self.count = len(self.names)


    def __getitem__(self, index):
        name = self.names[index]
        rayed = torch.Tensor()
        if (os.path.exists((os.path.join(self.data_root, '/labels')))):
            img = Image.open((os.path.join(self.data_root, '/images/'+name)))
            rayed = Image.open((os.path.join(self.data_root, '/labels/'+name)))
        else:
            if(self.args.setup==4 or self.args.setup==5):
                img = Image.fromarray(np.load((os.path.join(self.data_root, name))))
            else:
                img =  Image.open((os.path.join(self.data_root, name)))
        img = self.transform(img)
        # img = (img-img.min())/(img.max()-img.min())
        return (rayed, img)

    def __len__(self):
        return self.count


##
## Loading data for training/testing
##
def load_data(args,test=False):
    data_train_loader, data_valid_loader, data_test_loader = [],[],[]
    if (not test):
        data_train = MyCustomDataset(args.dataperc,args.data_path+'/train', transform=transforms.Compose([
                                                                                  transforms.Resize((config.size, config.size)),
                                                                                  transforms.ToTensor()
                                                                                  ]),args=args)

        data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)


        data_valid = MyCustomDataset(-args.valid,args.data_path+'/valid',transform=transforms.Compose([
                                                                                  transforms.Resize((config.size, config.size)),
                                                                                  transforms.ToTensor()
                                                                                  ]),args=args)

        data_valid_loader = DataLoader(data_valid, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    if (test):
        data_test = MyCustomDataset(args.dataperc,args.data_path+'/test', transform=transforms.Compose([
                                                                                  transforms.Resize((config.size, config.size)),
                                                                                  transforms.ToTensor()
                                                                                  ]),args=args)

        data_test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    return (data_train_loader, data_valid_loader, data_test_loader)


##
## Creating noisy version/scans after Radon from the truth
##
def create(truths, mean):
    if(config.angles != 0):rayed = config.fwd_op_mod(truths)
    else: rayed=truths.clone()

    rayed += Variable(config.noise * mean * torch.randn(rayed.shape)).type_as(rayed)
    return rayed
