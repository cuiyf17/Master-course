# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pickle
from alphasim import AlphaBase
from alphasim import DataRegistry as dr
from alphasim import Oputil

# 将其他路径添加到sys.path中
sys.path.append(os.path.abspath('/home/cuiyf/myalphasim/'))
import cuiyf_op.cuiyfOp as OP


class Alpha(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndays = cfg.getAttributeDefault('ndays', 21)
        self.lag = cfg.getAttributeDefault('lag', 0)
        self.cap = dr.getData('cap')
        self.close = dr.getData('adj_close')
        self.status = dr.getData('status')
        self.volume = dr.getData('adj_volume')
        self.risk_free_return = 0
        self.pre_alpha = None
        self.tsrank = None

    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        returns = self.close[di - self.delay, valid_idx]/self.close[di - self.delay - 21, valid_idx] - 1
        group = OP.group_split2(returns, 5)

        returns1 = self.close[start_di:end_di, valid_idx]/self.close[start_di-1:end_di-1, valid_idx] - 1
        returns2 = self.close[start_di-1:end_di - 1, valid_idx]/self.close[start_di-2:end_di-2, valid_idx] - 1

        returns_skew = OP.skew(returns1)
        
        autocorr = OP.corr(returns1, returns2)
        
        alpha = OP.rank(autocorr)*OP.rank(-returns_skew)
        # alpha = OP.rank(alpha) - 0.5
        # alpha = OP.power(alpha, 3)
        # alpha = OP.groupNeutralize(alpha, group)
        


        if(self.pre_alpha is None):
            self.alpha[valid_idx] = alpha
            self.pre_alpha = self.alpha.copy()
        else:
            mask1 = ~np.isnan(self.pre_alpha[valid_idx])
            mask2 = ~np.isnan(self.pre_alpha)
            if((di - self.delay)%1 == 0):
                self.alpha[valid_idx&mask2] = 0.6*self.pre_alpha[valid_idx&mask2] + 0.4*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()
            else:
                self.alpha[valid_idx&mask2] = 1*self.pre_alpha[valid_idx&mask2] + 0*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()

    def checkpointSave(self, fh):
        pickle.dump(self.pre_alpha, fh)

    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)