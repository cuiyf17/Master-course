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
        self.ndays = cfg.getAttributeDefault('ndays', 10)
        self.lag = cfg.getAttributeDefault('lag', 0)
        self.status = dr.getData('status')
        self.cap = dr.getData('cap')
        self.negcap = dr.getData('negcap')
        self.close = dr.getData('adj_close')
        self.volume = dr.getData('adj_volume')
        self.vwap = dr.getData('adj_vwap')
        self.high = dr.getData('adj_high')
        self.low = dr.getData('adj_low')
        self.open = dr.getData('adj_open')
        self.amount = dr.getData('amount')
        self.industry = dr.getData('subindustry')
        self.risk_free_return = 0
        self.pre_alpha = None
        self.std_returns1 = None
        self.std_returns2 = None
        self.KK = None


    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        g_returns = self.close[di-self.delay, valid_idx]/self.close[di-self.delay-21, valid_idx] - 1
        g_close = self.close[di-self.delay, valid_idx]
        group1 = OP.group_split2(g_returns, 10)
        group2 = OP.group_split2(g_close, 10)

        returns = self.close[start_di:end_di, valid_idx]/self.close[start_di-1:end_di-1, valid_idx] - 1
        high = self.high[start_di:end_di, valid_idx]
        low = self.low[start_di:end_di, valid_idx]
        alpha = OP.mean(high/low-1)
        alpha = (alpha - np.nanmean(alpha))**2

        alpha = -OP.rank(alpha) * OP.moment(returns, 4)
        # alpha = OP.rank(alpha) - 0.5
        # alpha = OP.power(alpha, 2)
        alpha = OP.groupNeutralize(alpha, group2, standardization=True)
        alpha = OP.groupNeutralize(alpha, group1, standardization=False)

        if(self.pre_alpha is None):
            self.alpha[valid_idx] = alpha
            self.pre_alpha = self.alpha.copy()
        else:
            mask1 = ~np.isnan(self.pre_alpha[valid_idx])
            mask2 = ~np.isnan(self.pre_alpha)
            if((di - self.delay)%1 == 0):
                self.alpha[valid_idx&mask2] = 0.8*self.pre_alpha[valid_idx&mask2] + 0.2*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()
            else:
                self.alpha[valid_idx&mask2] = 1*self.pre_alpha[valid_idx&mask2] + 0*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()

    def checkpointSave(self, fh):
        pickle.dump(self.pre_alpha, fh)
        pickle.dump(self.std_returns1, fh)
        pickle.dump(self.std_returns2, fh)


    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)
        self.std_returns1 = pickle.load(fh)
        self.std_returns2 = pickle.load(fh)