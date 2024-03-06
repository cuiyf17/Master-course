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
        self.lag = cfg.getAttributeDefault('lag',0)
        self.cap = dr.getData('cap')
        self.close = dr.getData('adj_close')
        self.status = dr.getData('status')
        self.volume = dr.getData('adj_volume')
        self.risk_free_return = 0
        self.pre_alpha = None
        self.rank_returns = None
        self.rank_price = None
        self.start_price = None

    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        returns = self.close[di - self.delay, valid_idx]/self.close[di - self.delay - 126, valid_idx] - 1
        returns1 = self.close[di-self.delay, valid_idx]/self.close[di-self.delay-1, valid_idx] - 1
        price = self.close[di-self.delay, valid_idx]
        #std_price = OP.std(self.close[di-self.delay-63+1:di-self.delay+1, valid_idx])
        if(self.start_price is None):
            self.start_price = np.zeros(self.close[di].shape[0]) + np.nan
            self.start_price[valid_idx] = price
        else:
            mask = np.isnan(self.start_price)
            tmp = np.zeros(self.close[di].shape[0]) + np.nan
            tmp[valid_idx] = price
            self.start_price[mask&valid_idx] = tmp[mask&valid_idx]
        group1 = OP.group_split2(price, 1)
        group2 = OP.group_split2(returns, 5)

        tmp1 = np.zeros((1, self.close[di].shape[0])) + np.nan
        tmp2 = np.zeros((1, self.close[di].shape[0])) + np.nan
        tmp1[:, valid_idx] = OP.groupNeutralize(returns1, group1)
        tmp2[:, valid_idx] = OP.groupNeutralize(price/self.start_price[valid_idx], group1)
        if(self.rank_returns is None):
            self.rank_returns = tmp1
            self.rank_price = tmp2
        elif(self.rank_returns.shape[0]<self.ndays):
            self.rank_returns = np.concatenate([self.rank_returns, tmp1], axis=0)
            self.rank_price = np.concatenate([self.rank_price, tmp2], axis=0)
        else:
            self.rank_returns = np.concatenate([self.rank_returns[1:], tmp1], axis=0)
            self.rank_price = np.concatenate([self.rank_price[1:], tmp2], axis=0)


        autocorr = OP.corr(self.rank_returns[:, valid_idx], self.rank_price[:, valid_idx])

        alpha = -autocorr
        # alpha = OP.groupNeutralize(alpha, group2)
        # alpha  = OP.rank(alpha)
        # alpha = OP.power(alpha, 2)


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
        pickle.dump(self.rank_returns, fh)
        pickle.dump(self.rank_price, fh)
        pickle.dump(self.start_price, fh)

    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)
        self.rank_returns = pickle.load(fh)
        self.rank_price = pickle.load(fh)
        self.start_price = pickle.load(fh)