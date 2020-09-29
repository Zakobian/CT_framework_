import odl
import torch
import config
import numpy as np
import BaseAlg


###
###
### TV-regularization. Based on github.com/adler-j/goettingen_dl_course_2018
###
###



def check_params(tau, sigmas,opnorms):
    sum_part = sum(sigma * opnorm ** 2
                   for sigma, opnorm in zip(sigmas, opnorms))
    check_value = tau * sum_part
    assert check_value < 4, 'value must be < 4, got {}'.format(check_value)



class Algorithm(BaseAlg.baseOpt):
    def __init__(self,args,data_loaders,path):
        super(Algorithm, self).__init__(args,data_loaders,path)
        self.alpha = 0.48#0.8
        self.tau = 1.5
        self.iterates=500
        self.grad = odl.Gradient(config.space)
        self.f = odl.solvers.IndicatorNonnegativity(config.space)
        if(config.angles != 0):
            self.op = config.fwd_op
        else: self.op = odl.IdentityOperator(config.space)

        self.L = [self.op, self.grad]
        grad_norm = 1.1 * odl.power_method_opnorm(self.grad, maxiter=20)
        fwd_op_norm = 1.1 * odl.power_method_opnorm(self.op, maxiter=20)
        opnorms = [fwd_op_norm, grad_norm]  # identity has norm 1

        c = 3.0 / (len(opnorms) * self.tau)
        self.sigmas = [c / opnorm ** 2 for opnorm in opnorms]
        check_params(self.tau, self.sigmas, opnorms)


    def output(self,scans,truth,iterates=-1,alpha=0):
        if(iterates==-1):
            iterates=self.iterates
        if(alpha!=0):
            self.alpha=alpha
        y = scans.cpu()
        x = torch.zeros((scans.shape[0],scans.shape[1],self.op.domain.shape[0],self.op.domain.shape[1])).type_as(y).numpy()
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                shape = y[i,j].shape
                g = [odl.solvers.L2NormSquared(self.op.range).translated(y[i,j]),
                     self.alpha * odl.solvers.L1Norm(self.grad.range)]
                z = config.space.zero()
                odl.solvers.douglas_rachford_pd(z, self.f, g, self.L, tau = self.tau, sigma = self.sigmas, niter=iterates)
                x[i,j] = z
        return torch.Tensor(x).cuda()
