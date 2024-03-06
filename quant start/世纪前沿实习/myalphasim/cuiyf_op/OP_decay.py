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



class cuiyfDecay(AlphaOpBase):
    def __init__(self, cfg):
        AlphaOpBase.__init__(self, cfg)
        self.days = cfg.getAttributeDefault('days', 3)
        self.tmp_alpha = None
        #self.iclose = dr.getData('interval5m.close')
        #self.iopen = dr.getData('interval5m.open')
        #self.iamount = dr.getData('interval5m.amo')
        #self.industry = dr.getData('industry')
        #self.SS2 = dr.getData('ZZ500')
        
        return

    def apply(self, di, alpha):
        if(self.tmp_alpha is None):
            self.tmp_alpha = alpha.reshape(1,-1).copy()
        else:
            if(self.tmp_alpha.shape[0] < self.days):
                self.tmp_alpha = np.concatenate((self.tmp_alpha, alpha.reshape(1,-1)), axis=0)
            else:
                self.tmp_alpha = np.concatenate((self.tmp_alpha[1:], alpha.reshape(1,-1)), axis=0)
            tmp_alpha = DECAYLINEAR(self.tmp_alpha)
            tmp_alpha[np.isnan(alpha)] = np.nan

            alpha[:] = tmp_alpha[:]

        return

    def checkpointSave(self, fh):
        pickle.dump(self.tmp_alpha, fh)

    def checkpointLoad(self, fh):
        self.tmp_alpha = pickle.load(fh)
