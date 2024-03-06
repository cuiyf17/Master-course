import os
import sys
import pdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from alphasim import AlphaOpBase
from alphasim import Oputil
from alphasim import DataRegistry as dr
from alphasim.data import Universe as uv

def DECAYLINEAR(A):
    inds = A.shape[0]
    cols = A.shape[1]
    w=np.arange(1,inds+1).repeat(cols).reshape((inds,cols))
    w = np.where(~np.isnan(A),w,np.nan)
    w=w/np.nansum(w,axis=0)
    return np.nansum(A*w,axis=0)



class cuiyfSaveAlphaValue(AlphaOpBase):
    def __init__(self, cfg):
        AlphaOpBase.__init__(self, cfg)
        self.days = cfg.getAttributeDefault('days', 3)
        self.tmp_alpha = None
        #self.iclose = dr.getData('interval5m.close')
        #self.iopen = dr.getData('interval5m.open')
        #self.iamount = dr.getData('interval5m.amo')
        #self.industry = dr.getData('industry')
        #self.SS2 = dr.getData('ZZ500')
        self.alphaname = cfg.getConfig().getparent().getparent().attrib['id']
        now_path = os.getcwd()
        if not os.path.exists(now_path + '/alpha_values'):
            os.mkdir(now_path + '/alpha_values')
        self.alpha_path = now_path + '/alpha_values/' + self.alphaname
        with open(self.alpha_path, 'w') as f:
            f.write('')
        
        return

    def apply(self, di, alpha):
        with open(self.alpha_path, 'a') as f:
            alphavalue = alpha[~np.isnan(alpha)]
            line = " ".join([str(i) for i in alphavalue])
            f.write(line + '\n')
        

        return

    def checkpointSave(self, fh):
        # pickle.dump(self.tmp_alpha, fh)
        pass

    def checkpointLoad(self, fh):
        # self.tmp_alpha = pickle.load(fh)
        pass
