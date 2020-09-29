import config
import BaseAlg


class Algorithm(BaseAlg.baseOpt):
    def __init__(self,args,data_loaders,path):
        super(Algorithm, self).__init__(args,data_loaders,path)
    def output(self,scans,truth):
        return config.fbp_op_mod(scans).clamp(min=0,max=1)
