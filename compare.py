import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import traceback
from IPython.display import display, clear_output
from main_parser import parser
import config
import torchvision.transforms as transforms
from odl.contrib import fom

args = parser.parse_args()
config.init(args)
torch.manual_seed(args.seed)

from data_load import load_data, create
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from skimage.measure import compare_ssim as ssim_sk
from skimage.measure import compare_psnr as psnr_sk

##
## Setting gpu settings
##
if(args.gpu != None):torch.cuda.set_device(args.gpu)

##
## Saves images in image_batches with the given titles.
## Copied from github.com/adler-j/goettingen_dl_course_2018
##
def show_image_matrix(image_batches,epoch, titles=None, indices=None, alg='', **kwargs):
    if indices is None:
        displayed_batches = image_batches
    else:
        displayed_batches = [batch[indices] for batch in image_batches]

    displayed_batches = [batch.data if isinstance(batch, Variable) else batch
                         for batch in displayed_batches]


    nrows = len(displayed_batches[0])
    ncols = len(displayed_batches)
    if titles is None:
        titles = [''] * ncols

    figsize = 8
    font_size=16
    fig, rows = plt.subplots(
        nrows, ncols, sharex=True, sharey=True,
        figsize=(ncols * figsize, figsize * nrows))

    if nrows == 1:
        rows = [rows]

    for i, row in enumerate(rows):
        if ncols == 1:
            row = [row]
        for j,(name, batch, ax) in enumerate(zip(titles, displayed_batches, row)):
            if i == 0:
                ax.set_title(name)
            ax.imshow(batch[i].squeeze(), **kwargs)
            ax.set_xlabel("PSNR = {:.4f} dB, SSIM = {:.4f}, HaarPSI = {:.4f}".format(psnr_sk(displayed_batches[0][i].squeeze().cpu().numpy(),displayed_batches[j][i].squeeze().cpu().numpy(),data_range=displayed_batches[0][i].cpu().numpy().max().item()-displayed_batches[0][i].cpu().numpy().min().item()),ssim_sk(displayed_batches[0][i].squeeze().cpu().numpy(),displayed_batches[j][i].squeeze().cpu().numpy(),data_range=displayed_batches[0][i].cpu().numpy().max().item()-displayed_batches[0][i].cpu().numpy().min().item()), fom.haarpsi(displayed_batches[j][i].squeeze().cpu().numpy(),displayed_batches[0][i].squeeze().cpu().numpy())), fontsize = font_size)

    plt.savefig(config.data_path+'figs/'+args.alg+str(args.setup)+'/'+str(epoch)+'.png', bbox_inches='tight')
    plt.close()

##
## Shows 10 examples of images from the validation set to demonstrate how well the algorithm works
##
def visual(alg_dict,data_loader,epoch):

    for i, (scans, truth) in enumerate(data_loader):
        if (scans.nelement() == 0):
            scans = create(truth,alg_dict['FBP'].noisemean)
        if i == 1:
            break

    test_images = Variable(truth.detach())
    test_data = Variable(scans.cuda())
    outputs=[test_images.cpu()]
    titles=['Truth']
    for alg_name,Algorithm in alg_dict.items():
        if hasattr(Algorithm,'net') and Algorithm.nograd:
            Algorithm.net.eval()
        outputs.append(Algorithm.output(test_data,test_images.cuda()).detach().cpu())
        titles.append(alg_name)
    show_image_matrix(outputs, epoch, titles, indices=slice(0, args.batch_size), clim=[0, 1],alg=args.alg,cmap='bone')

##
##Based on the number of log files calculates the experiment number
##
def cntexpir():
    if not os.path.exists(config.data_path+'logs/compare'):
        os.makedirs(config.data_path+'logs/compare')
    if (args.expir != -1):
        return args.expir
    exps = np.array([int(i[3:]) for i in (os.listdir(config.data_path+'logs/compare/'))])
    expir=1
    if (len(exps) != 0):
        expir=exps.max()+1
    print('Experiment '+str(expir))
    args.expir=expir
    return expir


def validate(alg_dict,data_loaders,expir):
    for (alg_name,Algorithm) in alg_dict.items():
        Algorithm.test()
        del Algorithm

##
## Opens all the algorithms and initialises them.
## Returns a dictionary of the algorithms in the form 'Name of algorithm': Library for the algorithm
##
def load_nets(alg_list,args,data_loaders):
    dict={}
    for alg in alg_list:
        alglib = __import__(alg)
        args.alg=alg
        dict[alg]=alglib.Algorithm(args,data_loaders,path=config.data_path+'nets/net_compare/')
        if hasattr(dict[alg],'net'):
            print(f'No of Parameters: {sum(p.numel() for p in dict[alg].net.parameters())}')
    args.alg='compare'
    return dict


##
## Save one picture to showcase the algorithms
##
def save_examples(alg_dict,data_loader,epoch):
    for i, (scans, truth) in enumerate(data_loader):
        if (scans.nelement() == 0):
            scans = create(truth,alg_dict['FBP'].noisemean)
        if i == 1:
            break

    test_images = Variable(truth.detach())
    test_data = Variable(scans.cuda())
    outputs=[test_images.cpu()]
    titles=['Truth']
    for alg_name,Algorithm in alg_dict.items():
        if hasattr(Algorithm,'net') and Algorithm.nograd:
            Algorithm.net.eval()
        outputs.append(Algorithm.output(test_data,test_images.cuda()).detach().cpu())
        titles.append(alg_name)
    for i in range(len(outputs)):
        plt.imshow((outputs[i][0,0,:,:]).numpy(),cmap='bone',vmin=0.0,vmax=1.0) #windowing: vmin=0.0,vmax=0.50
        plt.gcf().set_size_inches(5.0,5.0)
        plt.xticks([])
        plt.yticks([])
        print("{}:PSNR = {:.4f} dB, SSIM = {:.4f}".format(titles[i],psnr_sk(outputs[0][0,0,:,:].numpy(),outputs[i][0,0,:,:].numpy(),data_range=outputs[i][0,0,:,:].numpy().max().item()-outputs[i][0,0,:,:].numpy().min().item()),ssim_sk(outputs[0][0,0,:,:].numpy(),outputs[i][0,0,:,:].numpy(),data_range=outputs[i][0,0,:,:].numpy().max().item()-outputs[i][0,0,:,:].numpy().min().item())))
        plt.savefig(config.data_path+'figs/compare/'+titles[i]+'.png', bbox_inches='tight', transparent = False, pad_inches=0.1)





##
## Compare the algorithms provided in the alg_list
##
def main(alg_list=['FBP','LPD','ADR','CLAR0','TV']):#
    print('Data loaded')
    expir = cntexpir()
    args.alg='compare'
    if not os.path.exists(config.data_path+'figs/'+args.alg+str(args.setup)):
        os.makedirs(config.data_path+'figs/'+args.alg+str(args.setup))
    if not os.path.exists(config.data_path+'figs/'+args.alg+str(args.setup)+'/'+str(args.expir)):
        os.makedirs(config.data_path+'figs/'+args.alg+str(args.setup)+'/'+str(args.expir))
    data_loaders = load_data(args)
    alg_dict=load_nets(alg_list,args,data_loaders)
    save_examples(alg_dict,data_loaders[1],expir)
    visual(alg_dict,data_loaders[0],expir)
    validate(alg_dict,data_loaders,expir)


if __name__ == '__main__': main()
