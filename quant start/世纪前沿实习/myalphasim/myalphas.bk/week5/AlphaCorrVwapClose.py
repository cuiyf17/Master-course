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
        self.ndays = cfg.getAttributeDefault('ndays', 63)
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
        self.family = dr.getData('family')
        self.risk_free_return = 0
        self.pre_alpha = None


    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))
        #前一天涨跌停的股票不要
        #mask1 = (np.abs(self.close[di-self.delay]/self.close[di-self.delay-1]-1) > 0.198) & (self.family[di-self.delay]==2)
        #mask2 = (np.abs(self.close[di-self.delay]/self.close[di-self.delay-1]-1) > 0.099) & (self.family[di-self.delay]!=2)
        #valid_idx = valid_idx & ~mask1 & ~mask2


        industry = self.industry[di-self.delay, valid_idx]
        g_returns = self.close[di-self.delay, valid_idx]/self.close[di-self.delay-21, valid_idx] - 1
        g_close = self.close[di-self.delay, valid_idx]
        g_cap = self.cap[di-self.delay, valid_idx]
        g_amount = self.amount[di-self.delay, valid_idx]
        g_std = OP.std(self.close[start_di:end_di, valid_idx]/self.close[start_di-1:end_di-1, valid_idx] - 1)
        group1 = OP.group_split2(g_returns, 10)
        group2 = OP.group_split2(g_close, 10)
        group3 = OP.group_split2(g_cap, 10)

        vwap = self.vwap[start_di:end_di, valid_idx]
        close = self.close[start_di:end_di, valid_idx]

        alpha = OP.corr(vwap-close, close)
        alpha = OP.groupNeutralize(alpha, group1)
        # alpha = OP.groupNeutralize(alpha, group1, standardization=False)
        # alpha = -OP.rank(alpha)
        alpha = OP.power(alpha, 3)

        if(self.pre_alpha is None):
            self.alpha[valid_idx] = alpha
            self.pre_alpha = self.alpha.copy()
        else:
            mask1 = ~np.isnan(self.pre_alpha[valid_idx])
            mask2 = ~np.isnan(self.pre_alpha)
            if((di - self.delay)%1 == 0):
                self.alpha[valid_idx&mask2] = 0.5*self.pre_alpha[valid_idx&mask2] + 0.5*alpha[mask1]
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
