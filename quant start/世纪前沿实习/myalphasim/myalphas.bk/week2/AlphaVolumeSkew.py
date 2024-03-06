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
import cuiyf_op.cuiyfOp as cuiyfOp


class Alpha(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndays = cfg.getAttributeDefault('ndays', 63)
        self.lag = cfg.getAttributeDefault('lag', 0)
        self.cap = dr.getData('cap')
        self.close = dr.getData('adj_close')
        self.status = dr.getData('status')
        self.volume = dr.getData('adj_volume')
        self.risk_free_return = 0
        self.pre_alpha = None

    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))


        #returns = self.close[start_di:end_di, valid_idx] / self.close[start_di - 1:end_di - 1, valid_idx] - 1
        volume = self.volume[start_di:end_di, valid_idx]
        #cumvolume = cuiyfOp.sum(self.volume[start_di + self.ndays: end_di + self.lag, valid_idx])
        skew_vol = cuiyfOp.skew(self.volume[di - self.delay - 63 - 63 + 1:di - self.delay - 63 + 1, valid_idx])
        std_vol = cuiyfOp.std(self.volume[di - self.delay - 63 + 1:di - self.delay + 1, valid_idx])
        #mean_vol = cuiyfOp.mean(self.volume[di - self.delay - 63 + 1:di - self.delay + 1, valid_idx])
        #std_close = cuiyfOp.std(self.close[start_di + self.ndays: end_di + self.lag, valid_idx])
        
        alpha = cuiyfOp.rank(skew_vol)/(std_vol + 1)
        g_returns = self.close[di-self.delay, valid_idx]/self.close[di-self.delay-21, valid_idx]
        g_close = self.close[di-self.delay, valid_idx]
        g_std = cuiyfOp.std(self.close[di-self.delay-9:di-self.delay+1, valid_idx]/self.close[di-self.delay-9-1:di-self.delay+1-1, valid_idx])
        group1 = cuiyfOp.group_split2(g_returns, 10)
        group2 = cuiyfOp.group_split2(g_close, 10)
        group3 = cuiyfOp.group_split(g_std, 10)
        alpha = cuiyfOp.groupNeutralize(alpha, group2)
        alpha = cuiyfOp.power(alpha, 3)

        if(self.pre_alpha is None):
            self.alpha[valid_idx] = alpha
            self.pre_alpha = self.alpha.copy()
        else:
            mask1 = ~np.isnan(self.pre_alpha[valid_idx])
            mask2 = ~np.isnan(self.pre_alpha)
            if((di - self.delay)%21 == 0):
                self.alpha[valid_idx&mask2] = 0*self.pre_alpha[valid_idx&mask2] + 1*alpha[mask1]
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