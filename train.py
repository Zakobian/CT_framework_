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

    plt.savefig(config.data_path+'figs/'+args.alg+str(args.setup)+'/'+str(args.expir)+'/'+str(epoch)+'.png', bbox_inches='tight')
    plt.close()

##
## Shows 10 examples of images from the validation set to demonstrate how well the algorithm works
##
def visual(Algorithm,data_loader,epoch):
    if hasattr(Algorithm,'net') and Algorithm.nograd:
        Algorithm.net.eval()
    for i, (scans, truth) in enumerate(data_loader):
        if (scans.nelement() == 0):
            scans = create(truth,Algorithm.noisemean)
        if i == 1:
            break

    test_images = Variable(truth.detach())
    test_data = Variable(scans.cuda())
    outp=Algorithm.output(test_data,test_images.cuda()).detach().cpu()
    if(config.angles!=0):fbpp=config.fbp_op_mod(test_data).detach().cpu().clamp(min=0,max=1)
    else: fbpp=test_data.clone().cpu()

    alglib = __import__('TV')
    TV=alglib.Algorithm(args, (data_loader,data_loader,data_loader),'')
    print('starting TV')
    alpha=0.25#0.016;
    prevpsn=-1
    curpsn=0
    delt=2e-2
    TVrec=TV.output(test_data,test_images,iterates=500, alpha=alpha).cpu()
    diff=(torch.abs(fbpp-outp))
    for i in range(diff.shape[0]):
        diff[i]=(diff[i]-diff[i].min())/(diff[i].max()-diff[i].min())
    results = [test_images.cpu().permute(0,2,3,1),fbpp.permute(0,2,3,1), outp.permute(0,2,3,1),TVrec.permute(0,2,3,1)]
    titles = ['Truth','FBP',args.alg,'TV']

    show_image_matrix(results, epoch, titles, indices=slice(0, args.batch_size), clim=[0, 1],alg=args.alg,cmap='bone')

##
##Based on the number of log files calculates the experiment number
##
def cntexpir():
    if not os.path.exists(config.data_path+'logs/'+args.alg):
        os.makedirs(config.data_path+'logs/'+args.alg)
    if (args.expir != -1):
        return args.expir
    exps = np.array([int(i[3:]) for i in (os.listdir(config.data_path+'logs/'+args.alg+'/'))])
    expir=1
    if (len(exps) != 0):
        expir=exps.max()+1
    print('Experiment '+str(expir))
    args.expir=expir
    return expir

##
## Perform one cycle of train and test
##
def train_and_test(epoch,Algorithm,expir,data_loaders):
    if(not args.visual):
        writer = SummaryWriter(config.data_path+'logs/'+args.alg+'/exp'+str(expir)+'/train/'+str(epoch),comment='')
        Algorithm.train(writer,epoch)
        with SummaryWriter(config.data_path+'logs/'+args.alg+'/exp'+str(expir)+'/hypers',comment='Hype') as hypw:
            Algorithm.validate(writer,hypw,epoch)
        writer.close()
    visual(Algorithm,data_loaders[0],epoch)
    if(args.visual): exit()

def main():
    data_loaders = load_data(args)
    print('Data loaded')


    expir = cntexpir()
    if not os.path.exists(config.data_path+'figs/'+args.alg+str(args.setup)):
        os.makedirs(config.data_path+'figs/'+args.alg+str(args.setup))
    if not os.path.exists(config.data_path+'figs/'+args.alg+str(args.setup)+'/'+str(args.expir)):
        os.makedirs(config.data_path+'figs/'+args.alg+str(args.setup)+'/'+str(args.expir))


    alglib = __import__(args.alg)
    Algorithm=alglib.Algorithm(args,data_loaders)

    print(args.alg + ' net loaded in')

    ##
    ## Log the information about the current run
    ##
    with SummaryWriter(config.data_path+'logs/'+args.alg+'/exp'+str(expir)+'/',comment='') as w:
        w.add_text('Seed',str(args.seed))
        w.add_text('Setup',str(args.setup))
        w.add_text('Learning Rate',str(args.lr))
        w.add_text('Parameters',str(args))
        w.add_text('Data Percentage',str(args.dataperc))
        if hasattr(Algorithm,'net'):
            w.add_text('No of Parameters',str(sum(p.numel() for p in Algorithm.net.parameters())))
        if hasattr(Algorithm,'phi1net'):
            w.add_text('No of Parameters',str(sum(p.numel() for p in Algorithm.phi1net.parameters())))


    for ep in range(1, args.epochs+1):
        train_and_test(ep,Algorithm,expir,data_loaders)


if __name__ == '__main__': main()
