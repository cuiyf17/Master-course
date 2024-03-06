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
        self.cap = dr.getData('cap')
        self.close = dr.getData('adj_close')
        self.status = dr.getData('status')
        self.volume = dr.getData('adj_volume')
        self.risk_free_return = 0
        self.pre_alpha = None
        self.start_price = None

    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        g_returns = self.close[di-self.delay, valid_idx]/self.close[di-self.delay-21, valid_idx] - 1
        g_close = self.close[di-self.delay, valid_idx]
        group = OP.group_split2(g_returns, 5)

        price = self.close[di-self.delay, valid_idx]
        if(self.start_price is None):
            self.start_price = np.zeros(self.close[di].shape[0]) + np.nan
            self.start_price[valid_idx] = price
        else:
            mask = np.isnan(self.start_price)
            tmp = np.zeros(self.close[di].shape[0]) + np.nan
            tmp[valid_idx] = price
            self.start_price[mask&valid_idx] = tmp[mask&valid_idx]

        close = self.close[start_di:end_di, valid_idx]
        single_rets = close/self.close[start_di-1:end_di-1, valid_idx] - 1
        past_rets = np.nanmean(single_rets, axis=1).repeat(single_rets.shape[1]).reshape(single_rets.shape)
        single_rets = single_rets - past_rets
        single_index = close/self.start_price[valid_idx]
        past_index = np.nanmean(single_index, axis=1).repeat(single_index.shape[1]).reshape(single_index.shape)
        single_index = single_index - past_index
        #index = OP.groupNeutralize(single_index, group)
        #rets = OP.groupNeutralize(single_rets, group)
        
        corr = OP.corr(single_rets, single_index)
        alpha = -corr#*OP.moment(single_rets,4)
        #alpha = OP.groupNeutralize(alpha, group)
        # alpha = OP.rank(alpha) - 0.5
        # alpha = OP.power(alpha, 3)

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
        pickle.dump(self.start_price, fh)

    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)
        self.start_price = pickle.load(fh)