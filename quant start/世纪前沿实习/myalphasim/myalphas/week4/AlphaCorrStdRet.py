# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pickle
from alphasim import AlphaBase
from alphasim import DataRegistry as dr
from alphasim import Oputil
from alphasim import Universe as uv

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
        self.std_returns = None

    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        g_returns = self.close[di-self.delay, valid_idx]/self.close[di-self.delay-42, valid_idx] - 1
        g_close = self.close[di-self.delay, valid_idx]
        group1 = OP.group_split2(g_returns, 5)
        group2 = OP.group_split2(g_close, 10)

        returns = self.close[start_di:end_di, valid_idx] / self.close[start_di-1:end_di-1, valid_idx] - 1
        returns = returns# - np.nanmean(returns)
        std_returns = OP.std(returns)
        tmp = np.zeros((1, self.close[di].shape[0]))
        tmp[0, valid_idx] = std_returns
        if(self.std_returns is None):
            self.std_returns = tmp.copy()
        elif(self.std_returns.shape[0] < 21):
            self.std_returns = np.concatenate([self.std_returns, tmp], axis=0)
        else:
            self.std_returns = np.concatenate([self.std_returns[1:], tmp], axis=0)

        m = self.std_returns.shape[0]
        returns = self.close[di-self.delay-m+1:di-self.delay+1, valid_idx] / self.close[di-self.delay-m+1-21:di-self.delay+1-21, valid_idx] - 1
        returns = returns# - np.nanmean(returns, axis=1, keepdims=True)
        corr = OP.corr(returns, self.std_returns[:,valid_idx])
        #alpha = np.where((alpha > 1) & (alpha < 8), alpha, np.nan)
        alpha = -corr#*OP.moment(returns,4)

        # alpha = OP.power(alpha, 2)
        # alpha = OP.groupNeutralize(alpha, group1)
        # alpha = OP.groupNeutralize(alpha, group2)

        if(self.pre_alpha is None):
            self.alpha[valid_idx] = alpha
            self.pre_alpha = self.alpha.copy()
        else:
            mask1 = ~np.isnan(self.pre_alpha[valid_idx])
            mask2 = ~np.isnan(self.pre_alpha)
            if((di - self.delay)%1 == 0):
                self.alpha[valid_idx&mask2] = 0*self.pre_alpha[valid_idx&mask2] + 1*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()
            else:
                self.alpha[valid_idx&mask2] = 1*self.pre_alpha[valid_idx&mask2] + 0*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()

    def checkpointSave(self, fh):
        pickle.dump(self.pre_alpha, fh)
        pickle.dump(self.std_returns, fh)

    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)
        self.std_returns = pickle.load(fh)